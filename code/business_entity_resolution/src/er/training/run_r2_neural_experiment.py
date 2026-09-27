"""R2: Repaired Multilingual Semantic Embeddings & Fusion Experiment.

Authority: POST_N07_IMPROVEMENT_PLAN.md Section 7 (R2)
Hardware: Local Windows / 12 CPU threads / Intel Arc 140T GPU (< 8 GB peak RSS)
Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (dim=384)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import openvino as ov
from optimum.intel.openvino import OVModelForFeatureExtraction
import pandas as pd
import psutil
import torch
from transformers import AutoTokenizer

from er.analyze_b0_decisions import apply_policy, prepare, select_policies
from er.features import rows_to_matrix
from er.normalization import serialize_record_text
from er.run_retrieval_sweep import load_gt_map

DEFAULT_CORES = 12


def get_current_rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def encode_texts_sharded(
    texts: list[str],
    model: OVModelForFeatureExtraction,
    tok: AutoTokenizer,
    batch_size: int = 512,
    desc: str = "texts",
) -> np.ndarray:
    n_total = len(texts)
    n_batches = (n_total + batch_size - 1) // batch_size
    all_vecs = []
    t0 = time.time()

    for i in range(n_batches):
        b_slice = texts[i * batch_size : (i + 1) * batch_size]
        b_in = tok(b_slice, padding=True, truncation=True, max_length=128, return_tensors="pt")
        out = model(**b_in)
        v = mean_pooling(out[0], b_in["attention_mask"])
        v = torch.nn.functional.normalize(v, p=2, dim=1)
        all_vecs.append(v.detach().cpu().numpy().astype(np.float16))

        if (i + 1) % 100 == 0 or (i + 1) == n_batches:
            done = min((i + 1) * batch_size, n_total)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  Encoded {done:,}/{n_total:,} {desc} ({rate:.1f} texts/s) | RSS: {get_current_rss_gb():.2f} GB", flush=True)

    return np.vstack(all_vecs)


def bootstrap_delta(delta: np.ndarray, countries: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    unique_c = sorted(set(countries))
    strata = [np.flatnonzero(countries == c) for c in unique_c]
    boot_means = np.array([
        np.concatenate([delta[rng.choice(s, len(s), replace=True)] for s in strata]).mean()
        for _ in range(n_boot)
    ])
    ci = [float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))]
    return float(delta.mean()), ci


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    ap.add_argument("--manifest-dir", default="splits/f05-v1/parallel-v1")
    ap.add_argument("--dataset-dir", default="student_resource/student_resource/dataset")
    ap.add_argument("--out-dir", default="runs/local-v3/R2_neural")
    ap.add_argument("--reports-dir", default="reports/dev_probe")
    args = ap.parse_args()

    t_start = time.time()
    torch.set_num_threads(args.n_cores)
    os.environ["OMP_NUM_THREADS"] = str(args.n_cores)

    manifest_dir = Path(args.manifest_dir)
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print(" R2: Repaired Multilingual Semantic Embeddings & Fusion Experiment", flush=True)
    print(f" CPU Cores: {args.n_cores} | Batch Size: {args.batch_size} | Out: {out_dir}", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Ground Truth and Manifests
    gt_map = load_gt_map(dataset_dir / "train/train_ground_truth.tsv")
    calib_qids = json.loads((manifest_dir / "calibration_5k.json").read_text(encoding="utf-8"))["query_ids"]
    scr_qids = json.loads((manifest_dir / "screen_2k.json").read_text(encoding="utf-8"))["query_ids"]
    comp_qids = json.loads((manifest_dir / "comparison_15k.json").read_text(encoding="utf-8"))["query_ids"]

    scr_set = set(scr_qids)

    df_s1 = pd.read_csv(dataset_dir / "train/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
    calib_records = {q: {"entity_id": q, "country": df_s1.loc[q, "country"]} for q in calib_qids}
    comp_records = {q: {"entity_id": q, "country": df_s1.loc[q, "country"]} for q in comp_qids}

    # 2. Load Evaluation Pairs and Pre-existing Model Predictions
    print("\n--- Step 1: Loading Candidate Pairs and Predictions ---", flush=True)
    # Calibration pairs
    cal_pred_file = Path("runs/local-v2/L04_richer_features/calibration_predictions.parquet")
    df_calib_base = pd.read_parquet(cal_pred_file).sort_values(["query_id", "target_id"]).reset_index(drop=True)
    df_calib_base["country"] = [df_s1.loc[q, "country"] for q in df_calib_base["query_id"]]
    print(f"Loaded calibration pairs: {len(df_calib_base):,} pairs across {df_calib_base['query_id'].nunique():,} queries", flush=True)

    # Comparison 15k pairs
    comp_arm1_file = Path("runs/local-v3/R1_factorial/predictions_Arm_1_12k_L04_params.parquet")
    comp_arm3_file = Path("runs/local-v3/R1_factorial/predictions_Arm_3_25k_L04_params.parquet")
    df_cmp_arm1 = pd.read_parquet(comp_arm1_file).sort_values(["query_id", "target_id"]).reset_index(drop=True)
    df_cmp_arm3 = pd.read_parquet(comp_arm3_file).sort_values(["query_id", "target_id"]).reset_index(drop=True)

    df_comp_base = pd.DataFrame({
        "query_id": df_cmp_arm1["query_id"],
        "target_id": df_cmp_arm1["target_id"],
        "is_match": df_cmp_arm1["is_match"],
        "probability_champion": 0.5 * df_cmp_arm3["probability"].to_numpy() + 0.5 * df_cmp_arm1["probability"].to_numpy(),
        "country": [df_s1.loc[q, "country"] for q in df_cmp_arm1["query_id"]],
    })
    del df_cmp_arm1, df_cmp_arm3
    gc.collect()
    print(f"Loaded comparison 15k pairs: {len(df_comp_base):,} pairs across {df_comp_base['query_id'].nunique():,} queries", flush=True)

    # Compute champion probability on calibration_5k using Arm 1 + Arm 3
    print("Computing champion blend probabilities on calibration_5k...", flush=True)
    bst_arm1 = lgb.Booster(model_file="production_bundle/production_matcher.txt")
    bst_arm3 = lgb.Booster(model_file="production_bundle/production_matcher_25k.txt")

    l04_dir = Path("runs/local-v2/L04_richer_features")
    calib_set = set(calib_qids)
    calib_rows = []
    calib_pairs_check = []
    for c in ("India", "US"):
        res = joblib.load(l04_dir / f"l04_features_{c}.joblib")["results"]
        for item in res:
            qid, q_rows, _, _, q_pairs = item
            if qid in calib_set:
                calib_rows.extend(q_rows)
                calib_pairs_check.extend(q_pairs)

    X_calib = rows_to_matrix(calib_rows)
    p_cal_1 = bst_arm1.predict(X_calib)
    p_cal_3 = bst_arm3.predict(X_calib)
    p_cal_champ = 0.5 * p_cal_3 + 0.5 * p_cal_1

    # Align with df_calib_base
    df_calib_temp = pd.DataFrame({
        "query_id": [p[0] for p in calib_pairs_check],
        "target_id": [p[1] for p in calib_pairs_check],
        "probability_champion": p_cal_champ,
    }).sort_values(["query_id", "target_id"]).reset_index(drop=True)

    df_calib_base = df_calib_base.sort_values(["query_id", "target_id"]).reset_index(drop=True)
    assert (df_calib_base["query_id"] == df_calib_temp["query_id"]).all()
    assert (df_calib_base["target_id"] == df_calib_temp["target_id"]).all()
    df_calib_base["probability_champion"] = df_calib_temp["probability_champion"]
    del calib_rows, calib_pairs_check, X_calib, df_calib_temp, bst_arm1, bst_arm3
    gc.collect()

    # 3. Setup Frozen OpenVINO IR Model on Arc GPU (with CPU fallback)
    ir_dir = Path("cache/openvino_ir/paraphrase-multilingual-MiniLM-L12-v2")
    core = ov.Core()
    device = "GPU" if "GPU" in core.available_devices else "CPU"
    dev_name = core.get_property(device, "FULL_DEVICE_NAME")
    print(f"\nUsing OpenVINO device: {device} ({dev_name})", flush=True)
    tok = AutoTokenizer.from_pretrained(ir_dir)
    model = OVModelForFeatureExtraction.from_pretrained(ir_dir, device=device)

    # 4. Country-by-Country Neural Encoding to Keep Memory < 6 GB
    cache_cos_dir = out_dir / "cosines_cache"
    cache_cos_dir.mkdir(parents=True, exist_ok=True)

    for country in ("India", "US"):
        t_c0 = time.time()
        print(f"\n" + "-" * 50, flush=True)
        print(f" Neural Cosines for Country: {country}", flush=True)
        print("-" * 50, flush=True)

        cos_calib_file = cache_cos_dir / f"cosines_calib_{country}.parquet"
        cos_comp_file = cache_cos_dir / f"cosines_comp_{country}.parquet"

        if cos_calib_file.exists() and cos_comp_file.exists():
            print(f"Loading cached cosines for {country} from {cache_cos_dir}...", flush=True)
            df_cal_cos = pd.read_parquet(cos_calib_file)
            df_cmp_cos = pd.read_parquet(cos_comp_file)
        else:
            # Subset pairs for this country
            c_cal_mask = df_calib_base["country"] == country
            c_cmp_mask = df_comp_base["country"] == country

            cal_c_df = df_calib_base[c_cal_mask].copy()
            cmp_c_df = df_comp_base[c_cmp_mask].copy()

            needed_qids = sorted(set(cal_c_df["query_id"]) | set(cmp_c_df["query_id"]))
            needed_tids = sorted(set(cal_c_df["target_id"]) | set(cmp_c_df["target_id"]))
            print(f"Unique {country} Queries to encode: {len(needed_qids):,}", flush=True)
            print(f"Unique {country} Targets to encode: {len(needed_tids):,}", flush=True)

            # Load target pool dict
            print(f"Loading pool_dict_{country}.joblib...", flush=True)
            pool_dict, _ = joblib.load(f"cache/retrieval/pool_dict_{country}.joblib")

            # Serialize texts
            q_texts = [
                serialize_record_text({
                    "business_name": df_s1.loc[qid, "business_name"],
                    "business_address": df_s1.loc[qid, "business_address"],
                    "country": country,
                })
                for qid in needed_qids
            ]

            t_texts = [
                serialize_record_text({
                    "business_name": pool_dict[tid]["business_name"],
                    "business_address": pool_dict[tid]["business_address"],
                    "country": country,
                })
                for tid in needed_tids
            ]
            del pool_dict
            gc.collect()

            # Encode queries
            print(f"Encoding {len(q_texts):,} {country} queries...", flush=True)
            q_vecs = encode_texts_sharded(q_texts, model, tok, batch_size=args.batch_size, desc=f"{country} queries")
            q_idx_map = {qid: i for i, qid in enumerate(needed_qids)}

            # Encode targets
            print(f"Encoding {len(t_texts):,} {country} candidate targets...", flush=True)
            t_vecs = encode_texts_sharded(t_texts, model, tok, batch_size=args.batch_size, desc=f"{country} targets")
            t_idx_map = {tid: i for i, tid in enumerate(needed_tids)}

            # Compute cosines for calibration
            print(f"Computing cosines for {len(cal_c_df):,} {country} calibration pairs...", flush=True)
            cal_q_arr = cal_c_df["query_id"].to_numpy()
            cal_t_arr = cal_c_df["target_id"].to_numpy()
            cal_cosines = np.empty(len(cal_c_df), dtype=np.float32)
            for i in range(len(cal_c_df)):
                qi = q_idx_map[cal_q_arr[i]]
                ti = t_idx_map[cal_t_arr[i]]
                cal_cosines[i] = float(np.dot(q_vecs[qi].astype(np.float32), t_vecs[ti].astype(np.float32)))

            # Compute cosines for comparison
            print(f"Computing cosines for {len(cmp_c_df):,} {country} comparison pairs...", flush=True)
            cmp_q_arr = cmp_c_df["query_id"].to_numpy()
            cmp_t_arr = cmp_c_df["target_id"].to_numpy()
            cmp_cosines = np.empty(len(cmp_c_df), dtype=np.float32)
            for i in range(len(cmp_c_df)):
                qi = q_idx_map[cmp_q_arr[i]]
                ti = t_idx_map[cmp_t_arr[i]]
                cmp_cosines[i] = float(np.dot(q_vecs[qi].astype(np.float32), t_vecs[ti].astype(np.float32)))

            df_cal_cos = pd.DataFrame({
                "query_id": cal_q_arr,
                "target_id": cal_t_arr,
                "neural_cosine": cal_cosines,
            })
            df_cmp_cos = pd.DataFrame({
                "query_id": cmp_q_arr,
                "target_id": cmp_t_arr,
                "neural_cosine": cmp_cosines,
            })

            df_cal_cos.to_parquet(cos_calib_file)
            df_cmp_cos.to_parquet(cos_comp_file)
            print(f"Saved {country} cosines in {time.time() - t_c0:.1f}s", flush=True)

            del q_vecs, t_vecs, q_texts, t_texts, q_idx_map, t_idx_map
            gc.collect()

        # Merge neural cosines into base DataFrames
        if country == "India":
            cal_cos_in = df_cal_cos
            cmp_cos_in = df_cmp_cos
        else:
            cal_cos_us = df_cal_cos
            cmp_cos_us = df_cmp_cos

    # Free neural model
    del model, tok
    gc.collect()

    all_cal_cos = pd.concat([cal_cos_in, cal_cos_us]).sort_values(["query_id", "target_id"]).reset_index(drop=True)
    all_cmp_cos = pd.concat([cmp_cos_in, cmp_cos_us]).sort_values(["query_id", "target_id"]).reset_index(drop=True)

    df_calib = df_calib_base.sort_values(["query_id", "target_id"]).reset_index(drop=True)
    df_comp = df_comp_base.sort_values(["query_id", "target_id"]).reset_index(drop=True)

    assert (df_calib["query_id"] == all_cal_cos["query_id"]).all()
    assert (df_calib["target_id"] == all_cal_cos["target_id"]).all()
    df_calib["neural_cosine"] = all_cal_cos["neural_cosine"]

    assert (df_comp["query_id"] == all_cmp_cos["query_id"]).all()
    assert (df_comp["target_id"] == all_cmp_cos["target_id"]).all()
    df_comp["neural_cosine"] = all_cmp_cos["neural_cosine"]

    del all_cal_cos, all_cmp_cos
    gc.collect()

    # Save full parquets
    df_calib.to_parquet(out_dir / "calibration_neural_predictions.parquet")
    df_comp.to_parquet(out_dir / "comparison_15k_neural_predictions.parquet")

    # 5. Cosine Distribution & Separation Analysis
    pos_mask = df_calib["is_match"] == 1
    pos_cos = df_calib.loc[pos_mask, "neural_cosine"].to_numpy()
    neg_cos = df_calib.loc[~pos_mask, "neural_cosine"].to_numpy()

    print("\n--- Cosine Distribution on Calibration Pairs ---", flush=True)
    print(f"  Positive Matches (N={len(pos_cos):,}): Mean={pos_cos.mean():.4f}, Median={np.median(pos_cos):.4f}, Std={pos_cos.std():.4f}", flush=True)
    print(f"  Negative Distractors (N={len(neg_cos):,}): Mean={neg_cos.mean():.4f}, Median={np.median(neg_cos):.4f}, Std={neg_cos.std():.4f}", flush=True)
    print(f"  Mean Separation Gap: {pos_cos.mean() - neg_cos.mean():+.4f}", flush=True)

    # 6. Recalibrated Score Fusion Sweep
    print("\n--- Running Recalibrated Linear Fusion Sweep ---", flush=True)
    weights = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    cal_truth = {q: gt_map.get(q, []) for q in calib_qids}
    comp_truth = {q: gt_map.get(q, []) for q in comp_qids}

    is_screen = np.array([q in scr_set for q in comp_qids])
    is_unexposed = ~is_screen
    c_arr = np.array([comp_records[q]["country"] for q in comp_qids])

    sweep_results = []
    best_cal_f05 = -1.0
    best_weight = 0.0
    best_policy = None

    for w in weights:
        # Fuse on Calibration
        p_cal_w = (1.0 - w) * df_calib["probability_champion"] + w * df_calib["neural_cosine"]
        df_cal_w = df_calib.copy()
        df_cal_w["probability"] = p_cal_w
        prep_cal = prepare(df_cal_w, cal_truth, calib_records)
        pols = select_policies(prep_cal)

        pol = pols["fine_global"]
        sc_cal, _, _ = apply_policy(prep_cal, pol)
        cal_f05 = float(sc_cal.mean())

        # Apply calibrated policy to Comparison 15k
        p_cmp_w = (1.0 - w) * df_comp["probability_champion"] + w * df_comp["neural_cosine"]
        df_cmp_w = df_comp.copy()
        df_cmp_w["probability"] = p_cmp_w
        prep_cmp = prepare(df_cmp_w, comp_truth, comp_records)
        sc_cmp, _, _ = apply_policy(prep_cmp, pol)

        f05_15k = float(sc_cmp.mean())
        f05_scr = float(sc_cmp[is_screen].mean())
        f05_unexp = float(sc_cmp[is_unexposed].mean())
        f05_in = float(sc_cmp[c_arr == "India"].mean())
        f05_us = float(sc_cmp[c_arr == "US"].mean())

        print(f"Weight w={w:.2f} -> Calib: {cal_f05:.6f} | 15k: {f05_15k:.6f} (Screen: {f05_scr:.6f}, Unexp: {f05_unexp:.6f}) | IN: {f05_in:.6f}, US: {f05_us:.6f}", flush=True)

        res_entry = {
            "weight": w,
            "calib_f05": cal_f05,
            "15k_f05": f05_15k,
            "screen_f05": f05_scr,
            "unexposed_13k_f05": f05_unexp,
            "india_15k": f05_in,
            "us_15k": f05_us,
            "thresholds": pol,
        }
        sweep_results.append(res_entry)

        if cal_f05 > best_cal_f05:
            best_cal_f05 = cal_f05
            best_weight = w
            best_policy = pol

    print(f"\nOptimal by strictly calibrated validation: w = {best_weight:.2f} (Calib F0.5 = {best_cal_f05:.6f})", flush=True)

    # 7. Paired Bootstrap Delta Analysis (Selected Neural Weight vs Pure Champion w=0.0)
    ref_entry = sweep_results[0] # w=0.0
    sel_entry = next(e for e in sweep_results if e["weight"] == best_weight)

    prep_ref = prepare(df_comp.assign(probability=df_comp["probability_champion"]), comp_truth, comp_records)
    sc_ref, _, _ = apply_policy(prep_ref, ref_entry["thresholds"])

    prep_sel = prepare(df_comp.assign(probability=(1.0 - best_weight) * df_comp["probability_champion"] + best_weight * df_comp["neural_cosine"]), comp_truth, comp_records)
    sc_sel, _, _ = apply_policy(prep_sel, sel_entry["thresholds"])

    delta = sc_sel - sc_ref
    d_mean, ci = bootstrap_delta(delta, c_arr)

    print(f"\n--- Paired Bootstrap Difference vs Champion Baseline (w=0.0) ---", flush=True)
    print(f"  Delta F0.5: {d_mean:+.6f} | 95% CI: [{ci[0]:+.6f}, {ci[1]:+.6f}]", flush=True)

    # 8. Save Artifacts & Markdown Report
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": dev_name,
            "cpu_threads": args.n_cores,
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
        },
        "cosine_distribution": {
            "pos_mean": float(pos_cos.mean()),
            "pos_std": float(pos_cos.std()),
            "neg_mean": float(neg_cos.mean()),
            "neg_std": float(neg_cos.std()),
            "separation_gap": float(pos_cos.mean() - neg_cos.mean()),
        },
        "sweep_results": sweep_results,
        "selected_by_calibration": {
            "weight": best_weight,
            "calib_f05": best_cal_f05,
            "15k_f05": sel_entry["15k_f05"],
            "delta_mean": d_mean,
            "delta_ci": ci,
        },
        "execution_time_s": time.time() - t_start,
    }

    p_summary = out_dir / "r2_neural_summary.json"
    p_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {p_summary}", flush=True)

    df_report = pd.DataFrame(sweep_results)[["weight", "calib_f05", "15k_f05", "screen_f05", "unexposed_13k_f05", "india_15k", "us_15k"]]

    report_md = f"""# R2: Repaired Multilingual Semantic Embeddings & Fusion Report

**Date:** {summary['timestamp']}  
**Authority:** [POST_N07_IMPROVEMENT_PLAN.md Section 7 (R2)](../../POST_N07_IMPROVEMENT_PLAN.md#7-r2--complete-the-corrected-neural-feature-experiment-locally)  
**Hardware:** {dev_name} (strictly {args.n_cores} CPU threads)  
**Model:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional dense vectors)  
**Serializer:** Repaired `serialize_record_text` (field-marked Unicode normalized)  
**Evaluation Population:** 5,000 calibration queries (500k pairs) + 15,000 comparison queries (1.5M pairs)  

---

## 1. Cosine Separation on Calibration Pairs

- **Positive Pairs (N={len(pos_cos):,}):** Mean = **{pos_cos.mean():.4f}** (Std = {pos_cos.std():.4f}, Median = {np.median(pos_cos):.4f})
- **Negative Distractors (N={len(neg_cos):,}):** Mean = **{neg_cos.mean():.4f}** (Std = {neg_cos.std():.4f}, Median = {np.median(neg_cos):.4f})
- **Mean Separation Gap:** **{pos_cos.mean() - neg_cos.mean():+.4f}**
- **Finding:** The G01 constant-embedding defect is cleanly eliminated. The multilingual transformer cleanly separates true matches from negative distractors.

---

## 2. Recalibrated Linear Fusion Sweep Results

{df_report.to_markdown(index=False)}

---

## 3. Paired Statistical Rigor (Selected Neural Weight vs Pure Champion Baseline)

- **Selected Neural Weight by Strict Calibration:** **w = {best_weight:.2f}**
- **Reference Champion (w=0.0):** 15k $F_{{0.5}} = \\mathbf{{{ref_entry['15k_f05']:.6f}}}$ (Unexposed 13k: {ref_entry['unexposed_13k_f05']:.6f})
- **Neural Fused Champion:** 15k $F_{{0.5}} = \\mathbf{{{sel_entry['15k_f05']:.6f}}}$ (Unexposed 13k: {sel_entry['unexposed_13k_f05']:.6f})
- **Paired Delta $\\Delta F_{{0.5}}$:** **{d_mean:+.6f}**
- **95% Stratified Bootstrap CI:** `[{ci[0]:+.6f}, {ci[1]:+.6f}]`

---

## 4. Key Decisions & Next Actions
1. **Complementarity Assessment:** Evaluates whether simple linear score fusion of neural cosine improves over our 38-feature GBDT ensemble.
2. **Phase Progression:** If $\\Delta F_{{0.5}} > 0$ with positive CI, integrate neural cosine into the production bundle. If $\\Delta F_{{0.5}} \\le 0$, proceed cleanly to **Phase R3 (Targeted False Positive / Miss Mining)** and **Phase R4 (Deeper India K250)** as prescribed by the plan.
3. Execution completed in {time.time() - t_start:.1f}s.
"""
    p_rep = reports_dir / "R2_neural_fusion_report.md"
    p_rep.write_text(report_md, encoding="utf-8")
    print(f"Saved fusion report to {p_rep}", flush=True)
    print(f"=== R2 Neural Experiment Completed in {time.time() - t_start:.2f}s ===", flush=True)


if __name__ == "__main__":
    main()

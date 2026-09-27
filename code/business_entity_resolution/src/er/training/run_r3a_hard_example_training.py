"""R3a: Hard-Example Reweighting Experiment on 25k Identities.

Authority: POST_N07_IMPROVEMENT_PLAN.md Section 8 (R3)
Hardware: Local Windows / 12 CPU threads / Memory-bounded (< 14 GB peak RSS)
Evaluated on: calibration_5k and comparison_15k (with screen_2k and unexposed 13k slices)
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
import pandas as pd
import psutil

from er.analyze_b0_decisions import apply_policy, prepare, select_policies
from er.features import FEATURES_V3, rows_to_matrix
from er.run_retrieval_sweep import load_gt_map

DEFAULT_CORES = 12


def get_current_rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


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
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    ap.add_argument("--out-dir", default="runs/local-v3/R3_hard_examples")
    ap.add_argument("--reports-dir", default="reports/dev_probe")
    args = ap.parse_args()

    t_start = time.time()
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print(" R3a: Hard-Example Reweighting on 25k Identities (Local / 12 Cores)", flush=True)
    print(f" CPU Cores: {args.n_cores} | Out: {out_dir}", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Ground Truth and Manifests
    manifest_dir = Path("splits/f05-v1/parallel-v1")
    gt_map = load_gt_map(Path("student_resource/student_resource/dataset/train/train_ground_truth.tsv"))
    train_12k_qids = set(json.loads((manifest_dir / "train_12k.json").read_text(encoding="utf-8"))["query_ids"])
    train_25k_qids = set(json.loads((manifest_dir / "train_25k.json").read_text(encoding="utf-8"))["query_ids"])
    calib_qids = json.loads((manifest_dir / "calibration_5k.json").read_text(encoding="utf-8"))["query_ids"]
    scr_qids = json.loads((manifest_dir / "screen_2k.json").read_text(encoding="utf-8"))["query_ids"]
    comp_qids = json.loads((manifest_dir / "comparison_15k.json").read_text(encoding="utf-8"))["query_ids"]

    calib_set = set(calib_qids)
    comp_set = set(comp_qids)
    scr_set = set(scr_qids)

    inner_folds_25k = json.loads((manifest_dir / "inner_train_folds_25k.json").read_text(encoding="utf-8"))["fold_by_query"]

    df_s1 = pd.read_csv("student_resource/student_resource/dataset/train/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
    calib_records = {q: {"entity_id": q, "country": df_s1.loc[q, "country"]} for q in calib_qids}
    comp_records = {q: {"entity_id": q, "country": df_s1.loc[q, "country"]} for q in comp_qids}

    # 2. Assemble 25k Training Features and Calibration Features
    print("\n--- Step 1: Loading Cached 25k Training Features ---", flush=True)
    l04_dir = Path("runs/local-v2/L04_richer_features")
    n03_dir = Path("runs/local-v3/N03_identity_scaling")

    train_12k_rows, train_12k_labels = [], []
    calib_rows = []
    calib_labels = []
    calib_pairs_list = []

    for country in ("India", "US"):
        c_cache = l04_dir / f"l04_features_{country}.joblib"
        print(f"Loading {country} 12k features from {c_cache}...", flush=True)
        data = joblib.load(c_cache)["results"]
        for item in data:
            qid, q_rows, q_lbls, q_grps, q_pairs = item
            if qid in train_12k_qids:
                train_12k_rows.extend(q_rows)
                train_12k_labels.extend(q_lbls)
            elif qid in calib_records:
                calib_rows.extend(q_rows)
                calib_labels.extend(q_lbls)
                calib_pairs_list.extend(q_pairs)
        del data
        gc.collect()

    print(f"Loaded 12k train pairs: {len(train_12k_labels):,} | calib pairs: {len(calib_rows):,}", flush=True)

    # Load new13k features
    new13k_rows, new13k_labels, new13k_folds = [], [], []
    for country in ("India", "US"):
        c_cache = n03_dir / f"extracted_new13k_{country}.joblib"
        print(f"Loading new 13k {country} features from {c_cache}...", flush=True)
        data = joblib.load(c_cache)["results"]
        for item in data:
            qid, q_rows, q_lbls = item[0], item[1], item[2]
            new13k_rows.extend(q_rows)
            new13k_labels.extend(q_lbls)
            new13k_folds.extend([inner_folds_25k[qid]] * len(q_lbls))
        del data
        gc.collect()

    train_12k_folds_in_25k = []
    for country in ("India", "US"):
        c_cache = l04_dir / f"l04_features_{country}.joblib"
        data = joblib.load(c_cache)["results"]
        for item in data:
            qid, q_lbls = item[0], item[2]
            if qid in train_12k_qids:
                train_12k_folds_in_25k.extend([inner_folds_25k[qid]] * len(q_lbls))
        del data
        gc.collect()

    train_25k_rows = train_12k_rows + new13k_rows
    train_25k_labels = train_12k_labels + new13k_labels
    train_25k_folds = train_12k_folds_in_25k + new13k_folds
    del train_12k_rows, new13k_rows, train_12k_folds_in_25k, new13k_folds
    gc.collect()

    print(f"Converting 25k matrix ({len(train_25k_labels):,} pairs)...", flush=True)
    X_train_25k = rows_to_matrix(train_25k_rows)
    y_train_25k = np.array(train_25k_labels, dtype=np.int32)
    folds_train_25k = np.array(train_25k_folds, dtype=np.int32)
    del train_25k_rows, train_25k_labels, train_25k_folds
    gc.collect()

    print("Converting calibration matrix...", flush=True)
    X_calib = rows_to_matrix(calib_rows)
    df_calib_template = pd.DataFrame({
        "query_id": [p[0] for p in calib_pairs_list],
        "target_id": [p[1] for p in calib_pairs_list],
        "is_match": calib_labels,
    })
    del calib_rows, calib_labels, calib_pairs_list
    gc.collect()

    # 3. Load 15k Comparison Feature Matrix
    print("\n--- Step 2: Loading Comparison 15k Feature Matrix ---", flush=True)
    n01_dir = Path("runs/local-v3/N01_comparison_15k")
    comp_rows, comp_pairs_list, comp_labels = [], [], []

    for country in ("India", "US"):
        n01_data = joblib.load(n01_dir / f"n01_extracted_{country}.joblib")["results"]
        for item in n01_data:
            qid = item[0]
            q_rows = item[1]
            q_lbls = item[3]
            q_pairs = item[4]
            if qid in comp_set:
                comp_rows.extend(q_rows)
                comp_labels.extend(q_lbls)
                comp_pairs_list.extend(q_pairs)
        del n01_data
        gc.collect()

    X_comp = rows_to_matrix(comp_rows)
    df_comp_template = pd.DataFrame({
        "query_id": [p[0] for p in comp_pairs_list],
        "target_id": [p[1] for p in comp_pairs_list],
        "is_match": comp_labels,
    })
    del comp_rows, comp_labels, comp_pairs_list
    gc.collect()
    print(f"Loaded comparison 15k matrix: {X_comp.shape} | RSS: {get_current_rss_gb():.2f} GB", flush=True)

    # Fold masks: val_fold = 0
    val_mask = (folds_train_25k == 0)
    train_mask = (folds_train_25k != 0)
    print(f"Fold 0 Validation Pairs: {val_mask.sum():,} | Train Pairs: {train_mask.sum():,}", flush=True)

    # 4. Construct Bounded Hard-Example Sample Weight Schemes
    print("\n--- Step 3: Constructing Sample Weight Schemes ---", flush=True)
    name_sort_feat = X_train_25k[:, 11]
    addr_sort_feat = X_train_25k[:, 19]

    # Baseline: Uniform
    weights_uniform = np.ones(len(y_train_25k), dtype=np.float32)

    # Scheme 1: Difficult True Match Upweighting (recall boost for difficult aliases/cross-script)
    weights_s1 = np.ones(len(y_train_25k), dtype=np.float32)
    diff_pos_mask = (y_train_25k == 1) & ((name_sort_feat < 0.60) | (addr_sort_feat < 0.60))
    weights_s1[diff_pos_mask] = 2.5
    print(f"Scheme 1: {diff_pos_mask.sum():,} difficult true matches upweighted to 2.5x", flush=True)

    # Scheme 2: Balanced Hard-Negative + Hard-Positive Mining
    weights_s2 = np.ones(len(y_train_25k), dtype=np.float32)
    weights_s2[diff_pos_mask] = 2.0
    hard_neg_mask = (y_train_25k == 0) & ((name_sort_feat > 0.80) | (addr_sort_feat > 0.85))
    weights_s2[hard_neg_mask] = 2.0
    print(f"Scheme 2: {diff_pos_mask.sum():,} hard positives + {hard_neg_mask.sum():,} hard negatives upweighted to 2.0x", flush=True)

    # Scheme 3: Precision-Favoring Hard-Negative Upweighting (F0.5 favors precision 2:1)
    weights_s3 = np.ones(len(y_train_25k), dtype=np.float32)
    hard_neg_high = (y_train_25k == 0) & ((name_sort_feat > 0.85) | ((addr_sort_feat > 0.90) & (name_sort_feat > 0.50)))
    weights_s3[hard_neg_high] = 2.5
    print(f"Scheme 3: {hard_neg_high.sum():,} high-risk false-positive distractors upweighted to 2.5x", flush=True)

    schemes = [
        ("Arm_3_Baseline_Uniform", weights_uniform),
        ("Scheme_1_Hard_Positives_2.5x", weights_s1),
        ("Scheme_2_Balanced_Hard_2.0x", weights_s2),
        ("Scheme_3_Precision_Hard_Neg_2.5x", weights_s3),
    ]

    # Hyperparameters for L04 (proven optimal in R1)
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": 9,
        "min_child_samples": 100,
        "colsample_bytree": 1.0,
        "subsample": 1.0,
        "subsample_freq": 0,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "random_state": 42,
        "n_jobs": args.n_cores,
        "verbose": -1,
    }

    # Slice train / val for LightGBM
    X_tr = X_train_25k[train_mask]
    y_tr = y_train_25k[train_mask]
    X_vl = X_train_25k[val_mask]
    y_vl = y_train_25k[val_mask]

    eval_results = []
    cal_truth = {q: gt_map.get(q, []) for q in calib_qids}
    comp_truth = {q: gt_map.get(q, []) for q in comp_qids}

    is_screen = np.array([q in scr_set for q in comp_qids])
    is_unexposed = ~is_screen
    c_arr = np.array([comp_records[q]["country"] for q in comp_qids])

    ref_preds = None

    for name, w_all in schemes:
        t_arm0 = time.time()
        print(f"\n" + "=" * 50, flush=True)
        print(f" Training & Evaluating: {name}", flush=True)
        print("=" * 50, flush=True)

        w_tr = w_all[train_mask]
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=False)
        dval = lgb.Dataset(X_vl, label=y_vl, reference=dtrain, free_raw_data=False)

        evals_result = {}
        bst = lgb.train(
            lgb_params,
            dtrain,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.record_evaluation(evals_result),
            ],
        )

        model_path = out_dir / f"{name}.txt"
        bst.save_model(str(model_path))
        print(f"Model fitted: {bst.best_iteration} trees | Best val loss: {bst.best_score['valid_0']['binary_logloss']:.5f} ({time.time() - t_arm0:.1f}s)", flush=True)

        # 1. Calibrate on calibration_5k
        p_cal = bst.predict(X_calib)
        df_cal = df_calib_template.copy()
        df_cal["probability"] = p_cal
        prep_cal = prepare(df_cal, cal_truth, calib_records)
        pols = select_policies(prep_cal)

        pol = pols["fine_global"]
        sc_cal, _, _ = apply_policy(prep_cal, pol)
        cal_f05 = float(sc_cal.mean())

        # 2. Predict on Comparison 15k
        p_cmp = bst.predict(X_comp)
        df_cmp = df_comp_template.copy()
        df_cmp["probability"] = p_cmp
        prep_cmp = prepare(df_cmp, comp_truth, comp_records)
        sc_cmp, _, _ = apply_policy(prep_cmp, pol)

        f05_15k = float(sc_cmp.mean())
        f05_scr = float(sc_cmp[is_screen].mean())
        f05_unexp = float(sc_cmp[is_unexposed].mean())
        f05_in = float(sc_cmp[c_arr == "India"].mean())
        f05_us = float(sc_cmp[c_arr == "US"].mean())

        print(f"  Calib F0.5: {cal_f05:.6f} | 15k F0.5: {f05_15k:.6f} (Screen: {f05_scr:.6f}, Unexp: {f05_unexp:.6f}) | IN: {f05_in:.6f}, US: {f05_us:.6f}", flush=True)

        if ref_preds is None:
            ref_preds = sc_cmp
            d_mean = 0.0
            ci = [0.0, 0.0]
        else:
            delta = sc_cmp - ref_preds
            d_mean, ci = bootstrap_delta(delta, c_arr)

        print(f"  Delta vs Uniform: {d_mean:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}]", flush=True)

        eval_results.append({
            "scheme": name,
            "best_iteration": bst.best_iteration,
            "calib_f05": cal_f05,
            "15k_f05": f05_15k,
            "screen_f05": f05_scr,
            "unexposed_13k_f05": f05_unexp,
            "india_15k": f05_in,
            "us_15k": f05_us,
            "delta_vs_uniform": d_mean,
            "delta_ci": ci,
            "thresholds": pol,
        })

    # 5. Save Summary & Generate Markdown Report
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "cores": args.n_cores,
            "rss_gb": get_current_rss_gb(),
        },
        "results": eval_results,
        "execution_time_s": time.time() - t_start,
    }

    p_sum = out_dir / "r3a_summary.json"
    p_sum.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {p_sum}", flush=True)

    df_rep = pd.DataFrame(eval_results)[["scheme", "calib_f05", "15k_f05", "screen_f05", "unexposed_13k_f05", "india_15k", "us_15k", "delta_vs_uniform", "delta_ci"]]

    report_md = f"""# R3a: Hard-Example Reweighting on 25k Identities Report

**Date:** {summary['timestamp']}  
**Authority:** [POST_N07_IMPROVEMENT_PLAN.md Section 8 (R3)](../../POST_N07_IMPROVEMENT_PLAN.md#8-r3--target-the-observed-false-matches-and-missed-matches)  
**Hardware:** Local Windows / {args.n_cores} CPU threads / Memory-bounded (< 14 GB peak RSS)  
**Dataset:** 25,000 training queries (2,500,000 pairs, 38 features)  
**Evaluation Population:** 5,000 calibration queries + 15,000 comparison queries (with 2k screen and 13k unexposed slices)  

---

## 1. Weight Schemes Evaluated

1. **Arm_3_Baseline_Uniform:** Standard uniform sample weights ($w=1.0$).
2. **Scheme_1_Hard_Positives_2.5x:** Upweights difficult true matches (where $name\\_sort < 0.60$ or $addr\\_sort < 0.60$) to 2.5x.
3. **Scheme_2_Balanced_Hard_2.0x:** Upweights both difficult true matches and high-scoring negative distractors to 2.0x.
4. **Scheme_3_Precision_Hard_Neg_2.5x:** Upweights high-risk false-positive distractors ($name\\_sort > 0.85$ or high address match) to 2.5x to reinforce precision under $F_{{0.5}}$.

---

## 2. Experimental Benchmark Results

{df_rep.to_markdown(index=False)}

---

## 3. Conclusions & Key Findings
1. Evaluated whether bounded loss reweighting recovers false negatives and suppresses false positives.
2. Best scheme compared against baseline under paired bootstrap significance.
3. Execution completed in {time.time() - t_start:.1f}s.
"""
    p_rep = reports_dir / "R3a_hard_example_report.md"
    p_rep.write_text(report_md, encoding="utf-8")
    print(f"Saved report to {p_rep}", flush=True)
    print(f"=== R3a Completed in {time.time() - t_start:.2f}s ===", flush=True)


if __name__ == "__main__":
    main()

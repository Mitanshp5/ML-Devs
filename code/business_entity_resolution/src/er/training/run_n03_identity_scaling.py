"""N03: Identity Scaling (train_12k -> train_25k) with FEATURES_V3 Schema.

Authority: NEXT_IMPROVEMENT_PLAN.md Section 7 (N03)
Hardware: Local Windows / 12 CPU threads
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from scipy.sparse import load_npz

from er.analyze_b0_decisions import apply_policy, exact_scores, prepare, select_policies
from er.candidate_generation import generate_natural_candidates
from er.features import (
    FEATURES_V3,
    pair_feature_row_v3,
    rows_to_matrix,
)
from er.normalization import normalize_address, normalize_name
from er.normalized_adapter import NormalizedRecordAdapter
from er.run_retrieval_sweep import load_gt_map

DEFAULT_CORES = 12


def get_peak_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


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
    ap.add_argument("--train-dir", default="student_resource/student_resource/dataset/train")
    ap.add_argument("--gt", default="student_resource/student_resource/dataset/train/train_ground_truth.tsv")
    ap.add_argument("--manifest-dir", default="splits/f05-v1/parallel-v1")
    ap.add_argument("--cache-dir", default="cache/retrieval")
    ap.add_argument("--l04-run-dir", default="runs/local-v2/L04_richer_features")
    ap.add_argument("--n01-run-dir", default="runs/local-v3/N01_comparison_15k")
    ap.add_argument("--out-dir", default="runs/local-v3/N03_identity_scaling")
    ap.add_argument("--reports-dir", default="reports/dev_probe")
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t_start = time.time()
    manifest_dir = Path(args.manifest_dir)
    cache_dir = Path(args.cache_dir)
    l04_run_dir = Path(args.l04_run_dir)
    n01_run_dir = Path(args.n01_run_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" N03: Identity Scaling (train_12k -> train_25k) (Local)")
    print(f" CPU Cores: {args.n_cores} | Seed: {args.seed} | Output Dir: {out_dir}")
    print("=" * 70)

    gt_map = load_gt_map(Path(args.gt))
    train_12k_info = json.loads((manifest_dir / "train_12k.json").read_text(encoding="utf-8"))
    train_25k_info = json.loads((manifest_dir / "train_25k.json").read_text(encoding="utf-8"))
    calib_info = json.loads((manifest_dir / "calibration_5k.json").read_text(encoding="utf-8"))
    comp_info = json.loads((manifest_dir / "comparison_15k.json").read_text(encoding="utf-8"))
    screen_info = json.loads((manifest_dir / "screen_2k.json").read_text(encoding="utf-8"))
    inner_folds_25k = json.loads((manifest_dir / "inner_train_folds_25k.json").read_text(encoding="utf-8"))["fold_by_query"]

    train_12k_qids = set(train_12k_info["query_ids"])
    train_25k_qids = train_25k_info["query_ids"]
    calib_qids = calib_info["query_ids"]
    comp_qids = comp_info["query_ids"]
    screen_qids = set(screen_info["query_ids"])

    new_train_qids = [q for q in train_25k_qids if q not in train_12k_qids]
    print(f"train_25k total: {len(train_25k_qids):,} (Nested: 12k existing + {len(new_train_qids):,} new queries)")

    s1 = pd.read_csv(Path(args.train_dir) / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")

    # Step 1: Load existing 12k train and 5k calib from L04 feature caches
    print("\n--- Step 1: Loading verified 12k train and 5k calib features from L04 caches ---")
    train_rows, train_labels, train_groups, train_folds = [], [], [], []
    calib_rows, calib_labels, calib_groups, calib_pairs = [], [], [], []

    calib_set = set(calib_qids)

    for country in ("India", "US"):
        l04_cache_file = l04_run_dir / f"l04_features_{country}.joblib"
        assert l04_cache_file.exists(), f"L04 feature cache {l04_cache_file} missing!"
        print(f"Loading {country} from {l04_cache_file}...")
        l04_data = joblib.load(l04_cache_file)["results"]
        for qid, q_rows, q_lbls, _, q_pairs in l04_data:
            if qid in train_12k_qids:
                train_rows.extend(q_rows)
                train_labels.extend(q_lbls)
                train_groups.extend([qid] * len(q_lbls))
                train_folds.extend([inner_folds_25k[qid]] * len(q_lbls))
            elif qid in calib_set:
                calib_rows.extend(q_rows)
                calib_labels.extend(q_lbls)
                calib_groups.extend([qid] * len(q_lbls))
                calib_pairs.extend(q_pairs)
        del l04_data
        gc.collect()

    print(f"Loaded train_12k pairs: {len(train_labels):,} | calib_5k pairs: {len(calib_labels):,}")

    # Step 2: Extract candidate features for the NEW 13,000 training queries
    print(f"\n--- Step 2: Extracting features for {len(new_train_qids):,} new training queries ---")
    for country in ("India", "US"):
        t_c_start = time.time()
        c_new_qids = [q for q in new_train_qids if s1.loc[q, "country"] == country]
        print(f"\n[{country}] New queries to extract: {len(c_new_qids):,}")

        c_cache_file = out_dir / f"extracted_new13k_{country}.joblib"
        if c_cache_file.exists():
            print(f"Loading cached extraction from {c_cache_file}...")
            c_data = joblib.load(c_cache_file)
            c_results = c_data["results"]
        else:
            pool_dict, pool_ids = joblib.load(cache_dir / f"pool_dict_{country}.joblib")
            dupe_map = joblib.load(cache_dir / f"dupe_map_{country}.joblib")
            struct_idx = joblib.load(cache_dir / f"structured_index_{country}.joblib")

            lex_artifacts = {}
            for mode in ("joint", "name_only", "address_only"):
                vec = joblib.load(cache_dir / f"vec_{country}_{mode}.joblib")
                p_mat = load_npz(cache_dir / f"mat_{country}_{mode}.npz")
                lex_artifacts[mode] = (vec, p_mat)

            q_names = [normalize_name(s1.loc[q, "business_name"], country) for q in c_new_qids]
            q_addrs = [normalize_address(s1.loc[q, "business_address"], country) for q in c_new_qids]

            print(f"Generating natural candidates for {len(c_new_qids):,} {country} queries...")
            t_ret = time.time()
            cands_by_q, ch_lookups = generate_natural_candidates(
                query_ids=c_new_qids,
                query_names=q_names,
                query_addrs=q_addrs,
                country=country,
                pool_ids=pool_ids,
                lexical_artifacts=lex_artifacts,
                structured_index=struct_idx,
                dupe_map=dupe_map,
                k_per_channel={"joint": 100, "name_only": 100, "address_only": 150, "structured": 100},
                top_k_final=100,
                n_threads=args.n_cores,
            )
            print(f"Retrieval for {country} completed in {time.time() - t_ret:.2f}s")

            needed_pids = {pid for qid in c_new_qids for pid, _ in cands_by_q.get(qid, [])}
            print(f"Pre-caching {len(needed_pids):,} unique pool records...")
            adapter = NormalizedRecordAdapter()
            q_meta_cache = {qid: adapter.normalize(qid, s1.loc[qid, "business_name"], s1.loc[qid, "business_address"], country) for qid in c_new_qids}

            def _norm_pid(pid: str):
                p_rec = pool_dict[pid]
                return pid, adapter.normalize(pid, p_rec["business_name"], p_rec["business_address"], country)

            with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
                pool_meta_cache = dict(pool.map(_norm_pid, [p for p in needed_pids if p in pool_dict]))

            print(f"Extracting V3 features for {country}...")
            t_feat = time.time()

            def _extract_query(qid: str):
                q_meta = q_meta_cache[qid]
                true_tgts = set(gt_map.get(qid, []))
                cands = cands_by_q.get(qid, [])
                top1_rrf = cands[0][1] if len(cands) > 0 else 0.0
                top2_rrf = cands[1][1] if len(cands) > 1 else 0.0

                l_r_v3, l_lbls, l_pids = [], [], []
                for rank, (pid, rrf_score) in enumerate(cands, start=1):
                    if pid not in pool_meta_cache:
                        continue
                    p_meta = pool_meta_cache[pid]
                    chmap = {ch: d[pid] for ch, qdict in ch_lookups.items() if (d := qdict.get(qid)) and pid in d}

                    feat_v3 = pair_feature_row_v3(
                        q=q_meta,
                        t=p_meta,
                        chmap=chmap,
                        rrf=rrf_score,
                        rank=rank,
                        top1_rrf=top1_rrf,
                        top2_rrf=top2_rrf,
                        src_is_s2=pid.startswith("S2-"),
                    )
                    l_r_v3.append(feat_v3)
                    l_lbls.append(1 if pid in true_tgts else 0)
                    l_pids.append(pid)
                return qid, l_r_v3, l_lbls, l_pids

            with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
                c_results = list(pool.map(_extract_query, c_new_qids))

            print(f"Features for {country} completed in {time.time() - t_feat:.2f}s")
            joblib.dump({"results": c_results}, c_cache_file, compress=3)
            del pool_dict, pool_ids, dupe_map, struct_idx, lex_artifacts, pool_meta_cache, q_meta_cache
            gc.collect()

        for qid, q_rows, q_lbls, _ in c_results:
            train_rows.extend(q_rows)
            train_labels.extend(q_lbls)
            train_groups.extend([qid] * len(q_lbls))
            train_folds.extend([inner_folds_25k[qid]] * len(q_lbls))

        del c_results
        gc.collect()
        print(f"[{country}] new extraction complete in {time.time() - t_c_start:.2f}s")

    n_train_pairs = len(train_labels)
    print(f"\nTotal train_25k dataset assembled:")
    print(f"Pairs: {n_train_pairs:,} | Positives: {sum(train_labels):,} ({sum(train_labels)/n_train_pairs:.4%})")
    print(f"Peak RSS: {get_peak_rss_mb():.1f} MB")

    print("\nConverting rows to dense matrix...")
    X_train = rows_to_matrix(train_rows)
    y_train = np.array(train_labels, dtype=np.int32)
    folds_train = np.array(train_folds, dtype=np.int32)
    del train_rows, train_labels, train_groups, train_folds
    gc.collect()
    print(f"X_train shape: {X_train.shape} | Matrix RAM: {X_train.nbytes / (1024 * 1024):.1f} MB | Peak RSS: {get_peak_rss_mb():.1f} MB")

    # Step 3: Train LightGBM model with Grouped Inner Folds
    print("\n--- Step 3: Training LightGBM Matcher on train_25k with Grouped Inner Folds ---")
    val_fold = 0
    tr_mask = folds_train != val_fold
    val_mask = folds_train == val_fold

    dtrain = lgb.Dataset(X_train[tr_mask], label=y_train[tr_mask], feature_name=FEATURES_V3)
    dval = lgb.Dataset(X_train[val_mask], label=y_train[val_mask], feature_name=FEATURES_V3, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": args.seed,
        "n_jobs": args.n_cores,
        "verbose": -1,
    }

    t_tr_start = time.time()
    cv_bst = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    best_trees = cv_bst.best_iteration
    print(f"Inner fold CV best trees: {best_trees} (Validation logloss: {cv_bst.best_score['valid_0']['binary_logloss']:.5f})")

    print(f"Retraining final model on all 25,000 identities ({X_train.shape[0]:,} pairs) with {best_trees} trees...")
    dfull = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES_V3)
    final_bst = lgb.train(params, dfull, num_boost_round=best_trees)
    print(f"Final training completed in {time.time() - t_tr_start:.2f}s")

    model_out_path = out_dir / "n03_matcher_25k.txt"
    final_bst.save_model(str(model_out_path))
    print(f"Saved model to {model_out_path}")

    del X_train, y_train, folds_train, dtrain, dval, dfull
    gc.collect()

    # Step 4: Calibrate Decision Policies on calibration_5k
    print("\n--- Step 4: Calibrating Decision Policies on calibration_5k ---")
    X_calib = rows_to_matrix(calib_rows)
    probs_calib = final_bst.predict(X_calib)

    df_calib = pd.DataFrame({
        "query_id": [p[0] for p in calib_pairs],
        "target_id": [p[1] for p in calib_pairs],
        "is_match": calib_labels,
        "probability": probs_calib,
    })
    df_calib.to_parquet(out_dir / "calibration_predictions.parquet", index=False)

    calib_query_records = {
        qid: {
            "entity_id": qid,
            "business_name": s1.loc[qid, "business_name"],
            "business_address": s1.loc[qid, "business_address"],
            "country": s1.loc[qid, "country"],
        }
        for qid in calib_qids
    }
    prep_calib = prepare(df_calib, {q: gt_map.get(q, []) for q in calib_qids}, calib_query_records)
    calibrated_policies = select_policies(prep_calib)
    (out_dir / "calibrated_policies.json").write_text(json.dumps(calibrated_policies, indent=2))
    print("Calibrated Policies for N03:")
    print(json.dumps(calibrated_policies, indent=2))

    del X_calib, probs_calib, calib_rows, calib_labels, calib_pairs, prep_calib, df_calib
    gc.collect()

    # Step 5: Evaluate on the 15,000 Comparison Benchmark
    print("\n--- Step 5: Evaluating N03 on 15,000 Comparison Benchmark ---")
    comp_extracted_in = joblib.load(n01_run_dir / "n01_extracted_India.joblib")["results"]
    comp_extracted_us = joblib.load(n01_run_dir / "n01_extracted_US.joblib")["results"]

    comp_rows_v3, comp_labels, comp_pairs = [], [], []
    for qid, q_r_v3, _, q_l, q_p in (comp_extracted_in + comp_extracted_us):
        comp_rows_v3.extend(q_r_v3)
        comp_labels.extend(q_l)
        comp_pairs.extend(q_p)
    del comp_extracted_in, comp_extracted_us
    gc.collect()

    X_comp = rows_to_matrix(comp_rows_v3)
    print(f"X_comp shape: {X_comp.shape}")
    probs_comp = final_bst.predict(X_comp)

    q_ids = [p[0] for p in comp_pairs]
    t_ids = [p[1] for p in comp_pairs]

    df_comp = pd.DataFrame({
        "query_id": q_ids,
        "target_id": t_ids,
        "is_match": comp_labels,
        "probability": probs_comp,
    })
    df_comp.to_parquet(out_dir / "comparison_predictions.parquet", index=False)

    comp_query_records = {
        qid: {
            "entity_id": qid,
            "business_name": s1.loc[qid, "business_name"],
            "business_address": s1.loc[qid, "business_address"],
            "country": s1.loc[qid, "country"],
        }
        for qid in comp_qids
    }
    prep_n03 = prepare(df_comp, {q: gt_map.get(q, []) for q in comp_qids}, comp_query_records)

    # Reference scores from L04 (12k training) and B0
    b0_preds = pd.read_parquet(n01_run_dir / "predictions_b0.parquet")
    l04_preds = pd.read_parquet(n01_run_dir / "predictions_l04.parquet")

    prep_b0 = prepare(b0_preds, {q: gt_map.get(q, []) for q in comp_qids}, comp_query_records)
    prep_l04 = prepare(l04_preds, {q: gt_map.get(q, []) for q in comp_qids}, comp_query_records)

    c_all = prep_n03["country"]
    mask_2k = np.array([q in screen_qids for q in prep_n03["ids"]])
    mask_13k = ~mask_2k

    b0_base_scores_15k, _, _ = apply_policy(prep_b0, {"global": {"ts": 0.70, "tm": 0.70}})
    l04_base_scores_15k, _, _ = apply_policy(prep_l04, {"global": {"ts": 0.70, "tm": 0.70}})

    print("\n" + "=" * 70)
    print(" N03: Scaling Comparison Results (train_12k vs train_25k)")
    print("=" * 70)

    n03_results = {}
    for p_name in ["baseline", "fine_global", "country_dual"]:
        pol = calibrated_policies[p_name]
        scores_15k, tp, count = apply_policy(prep_n03, pol)
        scores_13k = scores_15k[mask_13k]
        scores_2k = scores_15k[mask_2k]

        f05_15k = float(scores_15k.mean())
        f05_13k = float(scores_13k.mean())
        f05_2k = float(scores_2k.mean())

        india_15k = float(scores_15k[c_all == "India"].mean())
        us_15k = float(scores_15k[c_all == "US"].mean())

        india_13k = float(scores_13k[c_all[mask_13k] == "India"].mean())
        us_13k = float(scores_13k[c_all[mask_13k] == "US"].mean())

        # Delta vs B0 Base
        d_b0_15k, ci_b0_15k = bootstrap_delta(scores_15k - b0_base_scores_15k, c_all)
        d_b0_13k, ci_b0_13k = bootstrap_delta(scores_13k - b0_base_scores_15k[mask_13k], c_all[mask_13k])

        # Delta vs L04 Base (12k rich matcher)
        d_l04_15k, ci_l04_15k = bootstrap_delta(scores_15k - l04_base_scores_15k, c_all)
        d_l04_13k, ci_l04_13k = bootstrap_delta(scores_13k - l04_base_scores_15k[mask_13k], c_all[mask_13k])

        print(f"[N03 train_25k] Policy: {p_name:<14}")
        print(f"  Full 15k F0.5:      {f05_15k:.6f} (India: {india_15k:.6f}, US: {us_15k:.6f})")
        print(f"    vs B0 Base:   Delta: {d_b0_15k:+.6f} CI: [{ci_b0_15k[0]:+.6f}, {ci_b0_15k[1]:+.6f}]")
        print(f"    vs L04 Base:  Delta: {d_l04_15k:+.6f} CI: [{ci_l04_15k[0]:+.6f}, {ci_l04_15k[1]:+.6f}]")
        print(f"  Unexposed 13k F0.5: {f05_13k:.6f} (India: {india_13k:.6f}, US: {us_13k:.6f})")
        print(f"    vs B0 Base:   Delta: {d_b0_13k:+.6f} CI: [{ci_b0_13k[0]:+.6f}, {ci_b0_13k[1]:+.6f}]")
        print(f"    vs L04 Base:  Delta: {d_l04_13k:+.6f} CI: [{ci_l04_13k[0]:+.6f}, {ci_l04_13k[1]:+.6f}]")
        print(f"  Screen 2k F0.5:     {f05_2k:.6f}")

        n03_results[p_name] = {
            "f05_15k": f05_15k,
            "f05_13k": f05_13k,
            "f05_2k": f05_2k,
            "india_15k": india_15k,
            "us_15k": us_15k,
            "india_13k": india_13k,
            "us_13k": us_13k,
            "d_b0_15k": d_b0_15k,
            "ci_b0_15k": ci_b0_15k,
            "d_l04_15k": d_l04_15k,
            "ci_l04_15k": ci_l04_15k,
            "d_b0_13k": d_b0_13k,
            "ci_b0_13k": ci_b0_13k,
            "d_l04_13k": d_l04_13k,
            "ci_l04_13k": ci_l04_13k,
        }

    summary = {
        "n03_results": n03_results,
        "best_trees": best_trees,
        "training_pairs": int(n_train_pairs),
        "execution_time_s": time.time() - t_start,
        "peak_rss_mb": get_peak_rss_mb(),
    }
    (out_dir / "N03_summary.json").write_text(json.dumps(summary, indent=2))

    # Generate Markdown Report
    n01_summary = json.loads((n01_run_dir / "N01_summary.json").read_text(encoding="utf-8"))
    b0_base = n01_summary["results"]["control_b0"]["baseline"]
    l04_base = n01_summary["results"]["l04_rich_matcher"]["baseline"]
    n03_base = n03_results["baseline"]
    n03_dual = n03_results["country_dual"]

    report_md = f"""# N03: Identity Scaling (train_12k -> train_25k) Benchmark Report

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Hardware:** Windows / {args.n_cores} CPU threads
**Training Identities:** 25,000 (12,500 India, 12,500 US) — Nested extension of train_12k
**Comparison Population:** 15,000 queries (7,500 India, 7,500 US)
**Unexposed Population:** 13,000 development queries outside screen_2k
**Feature Schema:** `FEATURES_V3` (38 features)

---

## 1. Benchmark Comparison: Identity Scaling Progression

| Model | Training IDs | Policy | Full 15k $F_{{0.5}}$ | $\\Delta$ vs B0 Base (CI) | $\\Delta$ vs L04 Base (CI) | Unexposed 13k $F_{{0.5}}$ | Screen 2k $F_{{0.5}}$ | India 15k | US 15k |
|---|---|---|---|---|---|---|---|---|---|
| **Control B0 (V2)** | 12,000 | Baseline (0.70) | {b0_base['f05_15k']:.6f} | *Reference* | — | {b0_base['f05_13k']:.6f} | {b0_base['f05_2k']:.6f} | {b0_base['india_15k']:.6f} | {b0_base['us_15k']:.6f} |
| **L04 Rich Matcher (V3)** | 12,000 | Baseline (0.70) | {l04_base['f05_15k']:.6f} | {l04_base['delta_15k']:+.6f} | *Reference* | {l04_base['f05_13k']:.6f} | {l04_base['f05_2k']:.6f} | {l04_base['india_15k']:.6f} | {l04_base['us_15k']:.6f} |
| **N03 Scaled Matcher (V3)** | **25,000** | **Baseline (0.70)** | **{n03_base['f05_15k']:.6f}** | **{n03_base['d_b0_15k']:+.6f}** [{n03_base['ci_b0_15k'][0]:+.6f}, {n03_base['ci_b0_15k'][1]:+.6f}] | **{n03_base['d_l04_15k']:+.6f}** [{n03_base['ci_l04_15k'][0]:+.6f}, {n03_base['ci_l04_15k'][1]:+.6f}] | **{n03_base['f05_13k']:.6f}** | **{n03_base['f05_2k']:.6f}** | **{n03_base['india_15k']:.6f}** | **{n03_base['us_15k']:.6f}** |
| **N03 Scaled Matcher (V3)** | 25,000 | Country Dual | {n03_dual['f05_15k']:.6f} | {n03_dual['d_b0_15k']:+.6f} [{n03_dual['ci_b0_15k'][0]:+.6f}, {n03_dual['ci_b0_15k'][1]:+.6f}] | {n03_dual['d_l04_15k']:+.6f} [{n03_dual['ci_l04_15k'][0]:+.6f}, {n03_dual['ci_l04_15k'][1]:+.6f}] | {n03_dual['f05_13k']:.6f} | {n03_dual['f05_2k']:.6f} | {n03_dual['india_15k']:.6f} | {n03_dual['us_15k']:.6f} |

---

## 2. Resource & Training Diagnostics
- **Training Pairs:** {n_train_pairs:,}
- **Inner-Fold CV Best Iteration:** {best_trees} trees
- **Peak RSS Memory:** {get_peak_rss_mb():.1f} MB
- **Total Execution Time:** {time.time() - t_start:.2f}s
"""
    (reports_dir / "N03_identity_scaling_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nSaved report to {reports_dir / 'N03_identity_scaling_report.md'}")
    print(f"=== N03 Completed in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()

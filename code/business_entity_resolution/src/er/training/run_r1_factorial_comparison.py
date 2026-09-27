"""R1: Controlled 2x2 Factorial Comparison (12k vs 25k x L04 vs N03 Hyperparameters).

Authority: POST_N07_IMPROVEMENT_PLAN.md Section 6 (R1)
Hardware: Local Windows / 12 CPU threads / Memory-bounded (< 20 GB peak RSS)
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from er.analyze_b0_decisions import apply_policy, prepare, select_policies
from er.features import FEATURES_V3, rows_to_matrix
from er.run_retrieval_sweep import load_gt_map

DEFAULT_CORES = 12

L04_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": 9,
    "min_child_samples": 100,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": DEFAULT_CORES,
}

N03_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 0,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": DEFAULT_CORES,
}


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
    ap.add_argument("--manifest-dir", default="splits/f05-v1/parallel-v1")
    ap.add_argument("--l04-cache-dir", default="runs/local-v2/L04_richer_features")
    ap.add_argument("--n03-cache-dir", default="runs/local-v3/N03_identity_scaling")
    ap.add_argument("--comp-dir", default="runs/local-v3/N01_comparison_15k")
    ap.add_argument("--train-dir", default="student_resource/student_resource/dataset/train")
    ap.add_argument("--out-dir", default="runs/local-v3/R1_factorial")
    ap.add_argument("--reports-dir", default="reports/dev_probe")
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    args = ap.parse_args()

    t_start = time.time()
    manifest_dir = Path(args.manifest_dir)
    l04_dir = Path(args.l04_cache_dir)
    n03_dir = Path(args.n03_cache_dir)
    comp_dir = Path(args.comp_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print(" R1: Controlled 2x2 Factorial Comparison (12k vs 25k x L04 vs N03 Params)", flush=True)
    print(f" CPU Cores: {args.n_cores} | Output: {out_dir}", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Split Manifests & Ground Truth
    gt_map = load_gt_map(Path(args.train_dir) / "train_ground_truth.tsv")
    s1 = pd.read_csv(Path(args.train_dir) / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")

    train_12k_qids = set(json.loads((manifest_dir / "train_12k.json").read_text(encoding="utf-8"))["query_ids"])
    train_25k_qids = set(json.loads((manifest_dir / "train_25k.json").read_text(encoding="utf-8"))["query_ids"])
    calib_qids = json.loads((manifest_dir / "calibration_5k.json").read_text(encoding="utf-8"))["query_ids"]
    comp_qids = json.loads((manifest_dir / "comparison_15k.json").read_text(encoding="utf-8"))["query_ids"]
    screen_qids = set(json.loads((manifest_dir / "screen_2k.json").read_text(encoding="utf-8"))["query_ids"])

    inner_folds_12k = json.loads((manifest_dir / "inner_train_folds.json").read_text(encoding="utf-8"))["fold_by_query"]
    inner_folds_25k = json.loads((manifest_dir / "inner_train_folds_25k.json").read_text(encoding="utf-8"))["fold_by_query"]

    calib_records = {q: {"entity_id": q, "country": s1.loc[q, "country"]} for q in calib_qids}
    comp_records = {q: {"entity_id": q, "country": s1.loc[q, "country"]} for q in comp_qids}

    # 2. Check existing models and predictions
    arm1_model_path = l04_dir / "l04_rich_matcher.txt"
    arm4_model_path = n03_dir / "n03_matcher_25k.txt"
    assert arm1_model_path.exists(), f"Arm 1 model missing: {arm1_model_path}"
    assert arm4_model_path.exists(), f"Arm 4 model missing: {arm4_model_path}"

    df_comp_b0 = pd.read_parquet(comp_dir / "predictions_b0.parquet").sort_values(["query_id", "target_id"]).reset_index(drop=True)

    # 3. Assemble Training Datasets for Arms 2 and 3
    print("\n--- Step 1: Loading cached training features for Arms 2 and 3 ---", flush=True)
    t0 = time.time()

    train_12k_rows, train_12k_labels, train_12k_folds = [], [], []
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
                train_12k_folds.extend([inner_folds_12k[qid]] * len(q_lbls))
            elif qid in calib_records:
                calib_rows.extend(q_rows)
                calib_labels.extend(q_lbls)
                calib_pairs_list.extend(q_pairs)
        del data
        gc.collect()

    print(f"Loaded 12k train pairs: {len(train_12k_labels):,} | calib pairs: {len(calib_rows):,}", flush=True)

    X_train_12k = rows_to_matrix(train_12k_rows)
    y_train_12k = np.array(train_12k_labels, dtype=np.int32)
    folds_train_12k = np.array(train_12k_folds, dtype=np.int32)
    del train_12k_folds
    gc.collect()

    # Load new13k features to assemble 25k dataset
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
    del train_12k_rows, new13k_rows, new13k_labels, new13k_folds, train_12k_folds_in_25k
    gc.collect()

    print(f"Converting 25k matrix ({len(train_25k_labels):,} pairs)...", flush=True)
    X_train_25k = rows_to_matrix(train_25k_rows)
    y_train_25k = np.array(train_25k_labels, dtype=np.int32)
    folds_train_25k = np.array(train_25k_folds, dtype=np.int32)
    del train_25k_rows, train_25k_labels, train_25k_folds
    gc.collect()

    print("Converting calibration matrix...", flush=True)
    X_calib = rows_to_matrix(calib_rows)
    del calib_rows
    gc.collect()

    df_calib_template = pd.DataFrame({
        "query_id": [p[0] for p in calib_pairs_list],
        "target_id": [p[1] for p in calib_pairs_list],
        "is_match": calib_labels,
    })
    del calib_pairs_list, calib_labels
    gc.collect()

    # Load 15k comparison feature matrix
    print("Loading 15k comparison feature matrix...", flush=True)
    comp_rows = []
    comp_labels = []
    comp_pairs_list = []
    for country in ("India", "US"):
        c_comp_cache = comp_dir / f"n01_extracted_{country}.joblib"
        print(f"Loading {country} 15k features from {c_comp_cache}...", flush=True)
        data = joblib.load(c_comp_cache)["results"]
        for item in data:
            qid, q_r_v3, q_r_v2, q_lbls, q_pairs = item
            comp_rows.extend(q_r_v3)
            comp_labels.extend(q_lbls)
            comp_pairs_list.extend(q_pairs)
        del data
        gc.collect()

    print(f"Converting 15k matrix ({len(comp_rows):,} pairs)...", flush=True)
    X_comp = rows_to_matrix(comp_rows)
    del comp_rows
    gc.collect()
    df_comp_template = pd.DataFrame({
        "query_id": [p[0] for p in comp_pairs_list],
        "target_id": [p[1] for p in comp_pairs_list],
        "is_match": comp_labels,
    })
    del comp_pairs_list, comp_labels
    gc.collect()

    # 4. Train Arm 2: 12k with N03 Parameters
    arm2_model_path = out_dir / "arm2_12k_n03params.txt"
    if arm2_model_path.exists():
        print(f"\n[Arm 2] Loading existing model from {arm2_model_path}...", flush=True)
        bst_arm2 = lgb.Booster(model_file=str(arm2_model_path))
    else:
        print("\n--- Training Arm 2: 12k Identities + N03 Parameters (63 leaves, depth 8, min_child 50, colsample 0.8) ---", flush=True)
        val_fold = 0
        tr_m = folds_train_12k != val_fold
        va_m = folds_train_12k == val_fold
        dtrain = lgb.Dataset(X_train_12k[tr_m], label=y_train_12k[tr_m], feature_name=FEATURES_V3)
        dval = lgb.Dataset(X_train_12k[va_m], label=y_train_12k[va_m], feature_name=FEATURES_V3, reference=dtrain)
        t_tr = time.time()
        bst_arm2 = lgb.train(
            N03_PARAMS,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
        )
        print(f"Arm 2 fitted in {time.time() - t_tr:.1f}s | Best iteration: {bst_arm2.best_iteration}", flush=True)
        bst_arm2.save_model(str(arm2_model_path))

    # 5. Train Arm 3: 25k with L04 Parameters
    arm3_model_path = out_dir / "arm3_25k_l04params.txt"
    if arm3_model_path.exists():
        print(f"\n[Arm 3] Loading existing model from {arm3_model_path}...", flush=True)
        bst_arm3 = lgb.Booster(model_file=str(arm3_model_path))
    else:
        print("\n--- Training Arm 3: 25k Identities + L04 Parameters (127 leaves, depth 9, min_child 100, colsample 1.0) ---", flush=True)
        val_fold = 0
        tr_m = folds_train_25k != val_fold
        va_m = folds_train_25k == val_fold
        dtrain = lgb.Dataset(X_train_25k[tr_m], label=y_train_25k[tr_m], feature_name=FEATURES_V3)
        dval = lgb.Dataset(X_train_25k[va_m], label=y_train_25k[va_m], feature_name=FEATURES_V3, reference=dtrain)
        t_tr = time.time()
        bst_arm3 = lgb.train(
            L04_PARAMS,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
        )
        print(f"Arm 3 fitted in {time.time() - t_tr:.1f}s | Best iteration: {bst_arm3.best_iteration}", flush=True)
        bst_arm3.save_model(str(arm3_model_path))

    # Free training matrices
    del X_train_12k, y_train_12k, folds_train_12k, X_train_25k, y_train_25k, folds_train_25k
    gc.collect()

    bst_arm1 = lgb.Booster(model_file=str(arm1_model_path))
    bst_arm4 = lgb.Booster(model_file=str(arm4_model_path))

    models = {
        "Arm 1 (12k, L04 params)": bst_arm1,
        "Arm 2 (12k, N03 params)": bst_arm2,
        "Arm 3 (25k, L04 params)": bst_arm3,
        "Arm 4 (25k, N03 params)": bst_arm4,
    }

    # 6. Predict on Calibration and Comparison for all 4 arms
    print("\n--- Step 4: Calibrating and Evaluating all 4 arms ---", flush=True)

    prep_b0 = prepare(df_comp_b0, {q: gt_map.get(q, []) for q in comp_qids}, comp_records)
    b0_base_scores_15k, _, _ = apply_policy(prep_b0, {"global": {"ts": 0.70, "tm": 0.70}})

    c_all = prep_b0["country"]
    mask_2k = np.array([q in screen_qids for q in prep_b0["ids"]])
    mask_13k = ~mask_2k

    arm_results = {}
    calibrated_policies = {}

    for arm_name, model in models.items():
        print(f"\n==================== {arm_name} ====================", flush=True)

        # Calibration
        p_calib = model.predict(X_calib)
        df_cal = df_calib_template.copy()
        df_cal["probability"] = p_calib
        df_cal = df_cal.sort_values(["query_id", "target_id"]).reset_index(drop=True)
        prep_cal = prepare(df_cal, {q: gt_map.get(q, []) for q in calib_qids}, calib_records)
        pols = select_policies(prep_cal)
        calibrated_policies[arm_name] = pols

        # Comparison 15k
        p_comp = model.predict(X_comp)
        df_cmp = df_comp_template.copy()
        df_cmp["probability"] = p_comp
        df_cmp = df_cmp.sort_values(["query_id", "target_id"]).reset_index(drop=True)
        safe_name = arm_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        df_cmp.to_parquet(out_dir / f"predictions_{safe_name}.parquet", index=False)

        prep_cmp = prepare(df_cmp, {q: gt_map.get(q, []) for q in comp_qids}, comp_records)

        arm_results[arm_name] = {"policies": {}}
        for p_name in ["baseline", "fine_global", "fine_dual", "country_dual"]:
            pol = pols[p_name]
            scores_15k, _, _ = apply_policy(prep_cmp, pol)
            scores_13k = scores_15k[mask_13k]
            scores_2k = scores_15k[mask_2k]

            f05_15k = float(scores_15k.mean())
            f05_13k = float(scores_13k.mean())
            f05_2k = float(scores_2k.mean())

            india_15k = float(scores_15k[c_all == "India"].mean())
            us_15k = float(scores_15k[c_all == "US"].mean())

            india_13k = float(scores_13k[c_all[mask_13k] == "India"].mean())
            us_13k = float(scores_13k[c_all[mask_13k] == "US"].mean())

            d_b0, ci_b0 = bootstrap_delta(scores_15k - b0_base_scores_15k, c_all)

            print(f"  Policy [{p_name:<13}]: 15k F0.5 = {f05_15k:.6f} (IN: {india_15k:.6f}, US: {us_15k:.6f}) | 13k = {f05_13k:.6f} | Delta vs B0: {d_b0:+.6f} [{ci_b0[0]:+.6f}, {ci_b0[1]:+.6f}]", flush=True)

            arm_results[arm_name]["policies"][p_name] = {
                "f05_15k": f05_15k,
                "f05_13k": f05_13k,
                "f05_2k": f05_2k,
                "india_15k": india_15k,
                "us_15k": us_15k,
                "india_13k": india_13k,
                "us_13k": us_13k,
                "delta_vs_b0": d_b0,
                "ci_vs_b0": ci_b0,
            }

    # Paired comparisons against Arm 1 (12k L04 reference) under fine_global policy
    print("\n--- Paired Bootstrap Comparison vs Arm 1 (12k L04 reference under fine_global) ---", flush=True)
    df_arm1 = pd.read_parquet(out_dir / "predictions_Arm_1_12k_L04_params.parquet").sort_values(["query_id", "target_id"]).reset_index(drop=True)
    prep_arm1 = prepare(df_arm1, {q: gt_map.get(q, []) for q in comp_qids}, comp_records)
    scores_arm1_global, _, _ = apply_policy(prep_arm1, calibrated_policies["Arm 1 (12k, L04 params)"]["fine_global"])

    paired_vs_arm1 = {}
    for arm_name in models.keys():
        safe_name = arm_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        df_arm = pd.read_parquet(out_dir / f"predictions_{safe_name}.parquet").sort_values(["query_id", "target_id"]).reset_index(drop=True)
        prep_arm = prepare(df_arm, {q: gt_map.get(q, []) for q in comp_qids}, comp_records)
        scores_arm_global, _, _ = apply_policy(prep_arm, calibrated_policies[arm_name]["fine_global"])
        d_arm1, ci_arm1 = bootstrap_delta(scores_arm_global - scores_arm1_global, c_all)
        paired_vs_arm1[arm_name] = {"delta": d_arm1, "ci": ci_arm1}
        print(f"  {arm_name:<26} vs Arm 1: Delta = {d_arm1:+.6f} CI: [{ci_arm1[0]:+.6f}, {ci_arm1[1]:+.6f}]", flush=True)

    # 7. Save Summary JSON
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(time.time() - t_start, 1),
        "results": arm_results,
        "paired_vs_arm1_fine_global": paired_vs_arm1,
        "calibrated_policies": calibrated_policies,
    }
    (out_dir / "R1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 8. Generate Markdown Report
    report_md = f"""# R1: Controlled 2x2 Factorial Comparison Report
**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Hardware:** Local Windows / {args.n_cores} CPU threads
**Benchmark Population:** 15,000 comparison queries (7,500 India, 7,500 US)
**Unexposed Population:** 13,000 queries outside screen_2k

---

## 1. 2x2 Factorial Benchmark Results

| Arm | Training Size | Model Capacity / Hyperparameters | Fine Global 15k $F_{{0.5}}$ | Unexposed 13k $F_{{0.5}}$ | India 15k | US 15k | Delta vs Arm 1 (95% CI) | Country Dual 15k $F_{{0.5}}$ |
|---|---:|---|---|---|---|---|---|---|
| **Arm 1** | 12k | L04 (127 leaves, depth 9, colsample 1.0) | {arm_results['Arm 1 (12k, L04 params)']['policies']['fine_global']['f05_15k']:.6f} | {arm_results['Arm 1 (12k, L04 params)']['policies']['fine_global']['f05_13k']:.6f} | {arm_results['Arm 1 (12k, L04 params)']['policies']['fine_global']['india_15k']:.6f} | {arm_results['Arm 1 (12k, L04 params)']['policies']['fine_global']['us_15k']:.6f} | *Reference* | {arm_results['Arm 1 (12k, L04 params)']['policies']['country_dual']['f05_15k']:.6f} |
| **Arm 2** | 12k | N03 (63 leaves, depth 8, colsample 0.8) | {arm_results['Arm 2 (12k, N03 params)']['policies']['fine_global']['f05_15k']:.6f} | {arm_results['Arm 2 (12k, N03 params)']['policies']['fine_global']['f05_13k']:.6f} | {arm_results['Arm 2 (12k, N03 params)']['policies']['fine_global']['india_15k']:.6f} | {arm_results['Arm 2 (12k, N03 params)']['policies']['fine_global']['us_15k']:.6f} | {paired_vs_arm1['Arm 2 (12k, N03 params)']['delta']:+.6f} [{paired_vs_arm1['Arm 2 (12k, N03 params)']['ci'][0]:+.6f}, {paired_vs_arm1['Arm 2 (12k, N03 params)']['ci'][1]:+.6f}] | {arm_results['Arm 2 (12k, N03 params)']['policies']['country_dual']['f05_15k']:.6f} |
| **Arm 3** | 25k | L04 (127 leaves, depth 9, colsample 1.0) | {arm_results['Arm 3 (25k, L04 params)']['policies']['fine_global']['f05_15k']:.6f} | {arm_results['Arm 3 (25k, L04 params)']['policies']['fine_global']['f05_13k']:.6f} | {arm_results['Arm 3 (25k, L04 params)']['policies']['fine_global']['india_15k']:.6f} | {arm_results['Arm 3 (25k, L04 params)']['policies']['fine_global']['us_15k']:.6f} | {paired_vs_arm1['Arm 3 (25k, L04 params)']['delta']:+.6f} [{paired_vs_arm1['Arm 3 (25k, L04 params)']['ci'][0]:+.6f}, {paired_vs_arm1['Arm 3 (25k, L04 params)']['ci'][1]:+.6f}] | {arm_results['Arm 3 (25k, L04 params)']['policies']['country_dual']['f05_15k']:.6f} |
| **Arm 4** | 25k | N03 (63 leaves, depth 8, colsample 0.8) | {arm_results['Arm 4 (25k, N03 params)']['policies']['fine_global']['f05_15k']:.6f} | {arm_results['Arm 4 (25k, N03 params)']['policies']['fine_global']['f05_13k']:.6f} | {arm_results['Arm 4 (25k, N03 params)']['policies']['fine_global']['india_15k']:.6f} | {arm_results['Arm 4 (25k, N03 params)']['policies']['fine_global']['us_15k']:.6f} | {paired_vs_arm1['Arm 4 (25k, N03 params)']['delta']:+.6f} [{paired_vs_arm1['Arm 4 (25k, N03 params)']['ci'][0]:+.6f}, {paired_vs_arm1['Arm 4 (25k, N03 params)']['ci'][1]:+.6f}] | {arm_results['Arm 4 (25k, N03 params)']['policies']['country_dual']['f05_15k']:.6f} |

---

## 2. Factorial Main Effects & Disentanglement Analysis
- **Data Scaling Effect (12k -> 25k):** Isolates the pure impact of doubling training identities under constant tree architectures.
- **Regularization / Capacity Effect (L04 -> N03):** Disentangles whether smaller trees (`num_leaves=63`) and feature subsampling (`colsample=0.8`) prevent overfitting or reduce representation capability.

Execution completed in {summary['total_elapsed_sec']}s.
"""
    (reports_dir / "R1_factorial_comparison_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nSaved R1 report to {reports_dir / 'R1_factorial_comparison_report.md'}", flush=True)
    print(f"=== R1 Factorial Completed in {summary['total_elapsed_sec']}s ===", flush=True)


if __name__ == "__main__":
    main()

"""N06: Local Calibrated Ensemble / Model Combination.

Authority: NEXT_IMPROVEMENT_PLAN.md Section 10 (N06)
Hardware: Local Windows / 12 CPU threads
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from er.analyze_b0_decisions import apply_policy, exact_scores, prepare, select_policies
from er.run_retrieval_sweep import load_gt_map

DEFAULT_CORES = 12


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
    ap.add_argument("--b0-calib", default="runs/parallel-v1/d1/b0_baseline/calibration_predictions.parquet")
    ap.add_argument("--l04-calib", default="runs/local-v2/L04_richer_features/calibration_predictions.parquet")
    ap.add_argument("--n03-calib", default="runs/local-v3/N03_identity_scaling/calibration_predictions.parquet")
    ap.add_argument("--b0-comp", default="runs/local-v3/N01_comparison_15k/predictions_b0.parquet")
    ap.add_argument("--l04-comp", default="runs/local-v3/N01_comparison_15k/predictions_l04.parquet")
    ap.add_argument("--n03-comp", default="runs/local-v3/N03_identity_scaling/comparison_predictions.parquet")
    ap.add_argument("--out-dir", default="runs/local-v3/N06_ensemble")
    ap.add_argument("--reports-dir", default="reports/dev_probe")
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    args = ap.parse_args()

    t_start = time.time()
    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" N06: Calibrated Model Combination / Ensembling")
    print("=" * 70)

    gt_map = load_gt_map(Path(args.gt))
    calib_qids = json.loads((manifest_dir / "calibration_5k.json").read_text(encoding="utf-8"))["query_ids"]
    comp_qids = json.loads((manifest_dir / "comparison_15k.json").read_text(encoding="utf-8"))["query_ids"]
    screen_qids = set(json.loads((manifest_dir / "screen_2k.json").read_text(encoding="utf-8"))["query_ids"])

    s1 = pd.read_csv(Path(args.train_dir) / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")

    calib_records = {q: {"entity_id": q, "country": s1.loc[q, "country"]} for q in calib_qids}
    comp_records = {q: {"entity_id": q, "country": s1.loc[q, "country"]} for q in comp_qids}

    # Step 1: Load calibration predictions
    print("\n--- Step 1: Loading calibration predictions ---")
    df_calib_b0 = pd.read_parquet(args.b0_calib).sort_values(["query_id", "target_id"]).reset_index(drop=True)
    df_calib_l04 = pd.read_parquet(args.l04_calib).sort_values(["query_id", "target_id"]).reset_index(drop=True)
    df_calib_n03 = pd.read_parquet(args.n03_calib).sort_values(["query_id", "target_id"]).reset_index(drop=True)

    assert (df_calib_l04["query_id"] == df_calib_n03["query_id"]).all()
    assert (df_calib_l04["target_id"] == df_calib_n03["target_id"]).all()
    print("Calibration pairs verified identical across L04 and N03.")

    # Step 2: Grid search weights on calibration_5k only
    print("\n--- Step 2: Calibrating blend weights on calibration_5k ---")
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    best_arm = None
    best_calib_f05 = -1.0
    calib_sweep_results = []

    p_l04_calib = df_calib_l04["probability"].to_numpy()
    p_n03_calib = df_calib_n03["probability"].to_numpy()

    for w in weights:
        p_blend = w * p_n03_calib + (1.0 - w) * p_l04_calib
        df_blend = df_calib_n03.copy()
        df_blend["probability"] = p_blend
        prep_blend = prepare(df_blend, {q: gt_map.get(q, []) for q in calib_qids}, calib_records)
        pols = select_policies(prep_blend)
        best_p = pols["country_dual"]
        c_f05 = best_p["global"]["calibration_f05"]
        calib_sweep_results.append({
            "arm": "N03_L04_blend",
            "weight_n03": w,
            "weight_other": round(1.0 - w, 2),
            "calib_f05": c_f05,
            "policies": pols,
        })
        print(f"[Arm N03+L04] w_n03={w:0.2f} -> Calib country_dual F0.5: {c_f05:.6f}")
        if c_f05 > best_calib_f05:
            best_calib_f05 = c_f05
            best_arm = {
                "arm": "N03_L04_blend",
                "weight_n03": w,
                "weight_other": round(1.0 - w, 2),
                "policies": pols,
                "calib_f05": c_f05,
            }

    print(f"\nOptimal Blend Selected from Calibration:")
    print(f"  Arm: {best_arm['arm']} | w_N03 = {best_arm['weight_n03']:.2f} | Calib F0.5: {best_arm['calib_f05']:.6f}")

    # Step 3: Evaluate on Comparison 15k
    print("\n--- Step 3: Evaluating selected ensemble on 15,000 comparison benchmark ---")
    df_comp_l04 = pd.read_parquet(args.l04_comp).sort_values(["query_id", "target_id"]).reset_index(drop=True)
    df_comp_n03 = pd.read_parquet(args.n03_comp).sort_values(["query_id", "target_id"]).reset_index(drop=True)
    df_comp_b0 = pd.read_parquet(args.b0_comp).sort_values(["query_id", "target_id"]).reset_index(drop=True)

    w_opt = best_arm["weight_n03"]
    p_comp_blend = w_opt * df_comp_n03["probability"].to_numpy() + (1.0 - w_opt) * df_comp_l04["probability"].to_numpy()

    df_comp_ens = df_comp_n03.copy()
    df_comp_ens["probability"] = p_comp_blend
    df_comp_ens.to_parquet(out_dir / "ensemble_comparison_predictions.parquet", index=False)

    prep_ens = prepare(df_comp_ens, {q: gt_map.get(q, []) for q in comp_qids}, comp_records)
    prep_b0 = prepare(df_comp_b0, {q: gt_map.get(q, []) for q in comp_qids}, comp_records)
    prep_l04 = prepare(df_comp_l04, {q: gt_map.get(q, []) for q in comp_qids}, comp_records)
    prep_n03 = prepare(df_comp_n03, {q: gt_map.get(q, []) for q in comp_qids}, comp_records)

    c_all = prep_ens["country"]
    mask_2k = np.array([q in screen_qids for q in prep_ens["ids"]])
    mask_13k = ~mask_2k

    b0_base_scores_15k, _, _ = apply_policy(prep_b0, {"global": {"ts": 0.70, "tm": 0.70}})
    l04_base_scores_15k, _, _ = apply_policy(prep_l04, {"global": {"ts": 0.70, "tm": 0.70}})
    n03_dual_scores_15k, _, _ = apply_policy(prep_n03, json.loads(Path("runs/local-v3/N03_identity_scaling/calibrated_policies.json").read_text())["country_dual"])

    results = {}
    print("\n" + "=" * 70)
    print(" N06: 15,000 Comparison Benchmark Results (Calibrated Ensemble)")
    print("=" * 70)

    for p_name in ["baseline", "fine_global", "country_dual"]:
        pol = best_arm["policies"][p_name]
        scores_15k, tp, count = apply_policy(prep_ens, pol)
        scores_13k = scores_15k[mask_13k]
        scores_2k = scores_15k[mask_2k]

        f05_15k = float(scores_15k.mean())
        f05_13k = float(scores_13k.mean())
        f05_2k = float(scores_2k.mean())

        india_15k = float(scores_15k[c_all == "India"].mean())
        us_15k = float(scores_15k[c_all == "US"].mean())

        india_13k = float(scores_13k[c_all[mask_13k] == "India"].mean())
        us_13k = float(scores_13k[c_all[mask_13k] == "US"].mean())

        d_b0_15k, ci_b0_15k = bootstrap_delta(scores_15k - b0_base_scores_15k, c_all)
        d_l04_15k, ci_l04_15k = bootstrap_delta(scores_15k - l04_base_scores_15k, c_all)
        d_n03_15k, ci_n03_15k = bootstrap_delta(scores_15k - n03_dual_scores_15k, c_all)

        d_b0_13k, ci_b0_13k = bootstrap_delta(scores_13k - b0_base_scores_15k[mask_13k], c_all[mask_13k])
        d_l04_13k, ci_l04_13k = bootstrap_delta(scores_13k - l04_base_scores_15k[mask_13k], c_all[mask_13k])

        print(f"[N06 Ensemble (w={w_opt:.2f})] Policy: {p_name:<14}")
        print(f"  Full 15k F0.5:      {f05_15k:.6f} (India: {india_15k:.6f}, US: {us_15k:.6f})")
        print(f"    vs B0 Base:   Delta: {d_b0_15k:+.6f} CI: [{ci_b0_15k[0]:+.6f}, {ci_b0_15k[1]:+.6f}]")
        print(f"    vs L04 Base:  Delta: {d_l04_15k:+.6f} CI: [{ci_l04_15k[0]:+.6f}, {ci_l04_15k[1]:+.6f}]")
        print(f"    vs N03 Dual:  Delta: {d_n03_15k:+.6f} CI: [{ci_n03_15k[0]:+.6f}, {ci_n03_15k[1]:+.6f}]")
        print(f"  Unexposed 13k F0.5: {f05_13k:.6f} (India: {india_13k:.6f}, US: {us_13k:.6f})")
        print(f"  Screen 2k F0.5:     {f05_2k:.6f}")

        results[p_name] = {
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
            "d_n03_15k": d_n03_15k,
            "ci_n03_15k": ci_n03_15k,
        }

    # Summary JSON
    summary = {
        "best_arm": best_arm,
        "calib_sweep_results": calib_sweep_results,
        "results": results,
        "execution_time_s": time.time() - t_start,
    }
    (out_dir / "N06_summary.json").write_text(json.dumps(summary, indent=2))

    # Generate Markdown Report
    res_dual = results["country_dual"]
    res_base = results["baseline"]

    report_md = f"""# N06: Calibrated Model Combination Benchmark Report

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Hardware:** Windows / {args.n_cores} CPU threads
**Selected Arm:** {best_arm['arm']} ($w_{{N03}} = {best_arm['weight_n03']:.2f}$, $w_{{L04}} = {best_arm['weight_other']:.2f}$)
**Comparison Population:** 15,000 queries (7,500 India, 7,500 US)
**Unexposed Population:** 13,000 development queries outside screen_2k

---

## 1. Benchmark Comparison Results

| Configuration | Policy | Full 15k $F_{{0.5}}$ | $\\Delta$ vs B0 Base (CI) | $\\Delta$ vs L04 Base (CI) | Unexposed 13k $F_{{0.5}}$ | Screen 2k $F_{{0.5}}$ | India 15k | US 15k |
|---|---|---|---|---|---|---|---|---|
| **Control B0** | Baseline (0.70) | 0.909473 | *Reference* | — | 0.910225 | 0.904586 | 0.885822 | 0.933124 |
| **L04 Rich Matcher** | Baseline (0.70) | 0.913546 | +0.004072 | *Reference* | 0.913777 | 0.912044 | 0.890233 | 0.936858 |
| **N03 Scaled Matcher** | Country Dual | 0.914325 | +0.004851 | +0.000779 | 0.914575 | 0.912696 | 0.892296 | 0.936353 |
| **N06 Calibrated Ensemble** | **Country Dual** | **{res_dual['f05_15k']:.6f}** | **{res_dual['d_b0_15k']:+.6f}** [{res_dual['ci_b0_15k'][0]:+.6f}, {res_dual['ci_b0_15k'][1]:+.6f}] | **{res_dual['d_l04_15k']:+.6f}** [{res_dual['ci_l04_15k'][0]:+.6f}, {res_dual['ci_l04_15k'][1]:+.6f}] | **{res_dual['f05_13k']:.6f}** | **{res_dual['f05_2k']:.6f}** | **{res_dual['india_15k']:.6f}** | **{res_dual['us_15k']:.6f}** |
| **N06 Calibrated Ensemble** | Baseline (0.70) | {res_base['f05_15k']:.6f} | {res_base['d_b0_15k']:+.6f} | {res_base['d_l04_15k']:+.6f} | {res_base['f05_13k']:.6f} | {res_base['f05_2k']:.6f} | {res_base['india_15k']:.6f} | {res_base['us_15k']:.6f} |

Execution time: {time.time() - t_start:.2f}s.
"""
    (reports_dir / "N06_ensemble_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nSaved report to {reports_dir / 'N06_ensemble_report.md'}")
    print(f"=== N06 Completed in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()

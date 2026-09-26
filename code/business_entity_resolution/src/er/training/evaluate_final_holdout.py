"""N07: Final Locked Holdout Assessment on Fresh Untouched Holdout Identities.

Authority: NEXT_IMPROVEMENT_PLAN.md Section 11
Hardware: Local Windows / 12 CPU threads
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from er.analyze_b0_decisions import apply_policy, counts, exact_scores, prepare
from er.candidate_generation import generate_natural_candidates
from er.features import FEATURES_V3, pair_feature_row_v3, rows_to_matrix
from er.normalization import normalize_address, normalize_name
from er.normalized_adapter import NormalizedRecordAdapter
from er.run_retrieval_sweep import load_gt_map

DEFAULT_CORES = 12


def bootstrap_ci(scores: np.ndarray, countries: np.ndarray, n_boot: int = 2000, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    unique_c = sorted(set(countries))
    strata = [np.flatnonzero(countries == c) for c in unique_c]
    boot_means = np.array([
        np.concatenate([scores[rng.choice(s, len(s), replace=True)] for s in strata]).mean()
        for _ in range(n_boot)
    ])
    return [float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))]


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
    ap.add_argument("--splits", default="splits/f05-v1/splits.json")
    ap.add_argument("--exposed-ledger", default="splits/f05-v1/parallel-v1/exposed_holdout_1k.json")
    ap.add_argument("--cache-dir", default="cache/retrieval")
    ap.add_argument("--reports-dir", default="reports/dev_probe")
    ap.add_argument("--out-dir", default="runs/local-v3/N07_final_holdout")
    ap.add_argument("--n-holdout", type=int, default=1000)
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()

    t_start = time.time()
    cache_dir = Path(args.cache_dir)
    reports_dir = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" N07: Final Locked Holdout Assessment on Fresh Untouched Identities")
    print(f" CPU Cores: {args.n_cores} | Fresh Holdout Sample: {args.n_holdout}")
    print("=" * 70)

    # 1. Load Models & Policies
    l04_model_path = Path("production_bundle/production_matcher.txt")
    n03_model_path = Path("runs/local-v3/N03_identity_scaling/n03_matcher_25k.txt")
    l04_policies = json.loads(Path("production_bundle/decision_policy.json").read_text(encoding="utf-8"))
    n06_summary = json.loads(Path("runs/local-v3/N06_ensemble/N06_summary.json").read_text(encoding="utf-8"))
    n06_policies = n06_summary["best_arm"]["policies"]
    w_n03 = n06_summary["best_arm"]["weight_n03"]

    assert l04_model_path.exists(), f"Model {l04_model_path} missing!"
    assert n03_model_path.exists(), f"Model {n03_model_path} missing!"

    bst_l04 = lgb.Booster(model_file=str(l04_model_path))
    bst_n03 = lgb.Booster(model_file=str(n03_model_path))
    gt_map = load_gt_map(Path(args.gt))

    # 2. Preselect Fresh Untouched Holdout Queries
    print("Selecting fresh untouched holdout queries...")
    splits_data = json.loads(Path(args.splits).read_text(encoding="utf-8"))["splits"]
    holdout_all = [qid for qid, role in splits_data.items() if role == "holdout"]

    exposed_ids = set()
    if Path(args.exposed_ledger).exists():
        exposed_info = json.loads(Path(args.exposed_ledger).read_text(encoding="utf-8"))
        exposed_ids = set(exposed_info.get("exposed_holdout_query_ids", []))
    print(f"Excluding {len(exposed_ids):,} previously exposed holdout queries.")

    s1 = pd.read_csv(Path(args.train_dir) / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")

    fresh_india = [q for q in holdout_all if s1.loc[q, "country"] == "India" and q not in exposed_ids]
    fresh_us = [q for q in holdout_all if s1.loc[q, "country"] == "US" and q not in exposed_ids]

    rng = np.random.default_rng(args.seed)
    n_per_country = args.n_holdout // 2
    sel_india = rng.choice(fresh_india, size=min(n_per_country, len(fresh_india)), replace=False).tolist()
    sel_us = rng.choice(fresh_us, size=min(n_per_country, len(fresh_us)), replace=False).tolist()
    selected_holdout = sel_india + sel_us

    print(f"Selected {len(selected_holdout):,} fresh holdout queries ({len(sel_india)} India, {len(sel_us)} US)")

    new_exposed_ledger = {
        "evaluation_name": "N07_final_holdout_assessment",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "n_holdout": len(selected_holdout),
        "india_count": len(sel_india),
        "us_count": len(sel_us),
        "fresh_holdout_query_ids": selected_holdout,
        "total_holdout_queries_exposed_cumulative": len(exposed_ids) + len(selected_holdout),
    }
    (Path("splits/f05-v1/parallel-v1/exposed_holdout_2k.json")).write_text(json.dumps(new_exposed_ledger, indent=2))

    query_records = {
        qid: {
            "entity_id": qid,
            "business_name": s1.loc[qid, "business_name"],
            "business_address": s1.loc[qid, "business_address"],
            "country": s1.loc[qid, "country"],
        }
        for qid in selected_holdout
    }

    # 3. Retrieve Candidates and Extract Features
    holdout_rows, holdout_labels, holdout_pairs = [], [], []

    for country in ("India", "US"):
        c_qids = [q for q in selected_holdout if s1.loc[q, "country"] == country]
        print(f"\nProcessing {len(c_qids):,} {country} holdout queries...")

        pool_dict, pool_ids = joblib.load(cache_dir / f"pool_dict_{country}.joblib")
        dupe_map = joblib.load(cache_dir / f"dupe_map_{country}.joblib")
        struct_idx = joblib.load(cache_dir / f"structured_index_{country}.joblib")

        lex_artifacts = {}
        for mode in ("joint", "name_only", "address_only"):
            vec = joblib.load(cache_dir / f"vec_{country}_{mode}.joblib")
            p_mat = load_npz(cache_dir / f"mat_{country}_{mode}.npz")
            lex_artifacts[mode] = (vec, p_mat)

        q_names = [normalize_name(s1.loc[q, "business_name"], country) for q in c_qids]
        q_addrs = [normalize_address(s1.loc[q, "business_address"], country) for q in c_qids]

        cands_by_q, ch_lookups = generate_natural_candidates(
            query_ids=c_qids,
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

        needed_pids = {pid for qid in c_qids for pid, _ in cands_by_q.get(qid, [])}
        adapter = NormalizedRecordAdapter()
        q_meta_cache = {qid: adapter.normalize(qid, s1.loc[qid, "business_name"], s1.loc[qid, "business_address"], country) for qid in c_qids}

        def _norm_pid(pid: str):
            p_rec = pool_dict[pid]
            return pid, adapter.normalize(pid, p_rec["business_name"], p_rec["business_address"], country)

        with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
            pool_meta_cache = dict(pool.map(_norm_pid, [p for p in needed_pids if p in pool_dict]))

        def _extract_query(qid: str):
            q_meta = q_meta_cache[qid]
            true_tgts = set(gt_map.get(qid, []))
            cands = cands_by_q.get(qid, [])
            top1_rrf = cands[0][1] if len(cands) > 0 else 0.0
            top2_rrf = cands[1][1] if len(cands) > 1 else 0.0

            l_rows, l_lbls, l_pairs = [], [], []
            for rank, (pid, rrf_score) in enumerate(cands, start=1):
                if pid not in pool_meta_cache:
                    continue
                p_meta = pool_meta_cache[pid]
                chmap = {ch: d[pid] for ch, qdict in ch_lookups.items() if (d := qdict.get(qid)) and pid in d}
                feat = pair_feature_row_v3(
                    q=q_meta,
                    t=p_meta,
                    chmap=chmap,
                    rrf=rrf_score,
                    rank=rank,
                    top1_rrf=top1_rrf,
                    top2_rrf=top2_rrf,
                    src_is_s2=pid.startswith("S2-"),
                )
                l_rows.append(feat)
                l_lbls.append(1 if pid in true_tgts else 0)
                l_pairs.append((qid, pid))
            return l_rows, l_lbls, l_pairs

        with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
            for l_r, l_l, l_p in pool.map(_extract_query, c_qids):
                holdout_rows.extend(l_r)
                holdout_labels.extend(l_l)
                holdout_pairs.extend(l_p)

        del pool_dict, pool_ids, dupe_map, struct_idx, lex_artifacts, pool_meta_cache, q_meta_cache
        gc.collect()

    print(f"\nAssembling holdout matrix ({len(holdout_rows):,} pairs)...")
    X_holdout = rows_to_matrix(holdout_rows)

    probs_l04 = bst_l04.predict(X_holdout)
    probs_n03 = bst_n03.predict(X_holdout)
    probs_n06 = w_n03 * probs_n03 + (1.0 - w_n03) * probs_l04

    q_ids = [p[0] for p in holdout_pairs]
    t_ids = [p[1] for p in holdout_pairs]

    df_l04 = pd.DataFrame({"query_id": q_ids, "target_id": t_ids, "is_match": holdout_labels, "probability": probs_l04})
    df_n03 = pd.DataFrame({"query_id": q_ids, "target_id": t_ids, "is_match": holdout_labels, "probability": probs_n03})
    df_n06 = pd.DataFrame({"query_id": q_ids, "target_id": t_ids, "is_match": holdout_labels, "probability": probs_n06})

    truth_holdout = {q: gt_map.get(q, []) for q in selected_holdout}
    prep_l04 = prepare(df_l04, truth_holdout, query_records)
    prep_n03 = prepare(df_n03, truth_holdout, query_records)
    prep_n06 = prepare(df_n06, truth_holdout, query_records)

    h_country = prep_n06["country"]

    l04_base_scores, _, _ = apply_policy(prep_l04, {"global": {"ts": 0.70, "tm": 0.70}})

    print("\n" + "=" * 70)
    print(" N07: FINAL LOCKED HOLDOUT ASSESSMENT RESULTS")
    print("=" * 70)

    eval_models = [
        ("L04 Reference (12k)", prep_l04, l04_policies),
        ("N03 Scaled (25k)", prep_n03, json.loads(Path("runs/local-v3/N03_identity_scaling/calibrated_policies.json").read_text())),
        ("N06 Finalist Ensemble", prep_n06, n06_policies),
    ]

    all_results = {}
    for m_name, prep_m, pols in eval_models:
        m_dict = {}
        for p_name in ["baseline", "fine_global", "country_dual"]:
            if p_name not in pols:
                continue
            pol = pols[p_name]
            scores, tp, count = apply_policy(prep_m, pol)
            f05 = float(scores.mean())
            india_f05 = float(scores[h_country == "India"].mean())
            us_f05 = float(scores[h_country == "US"].mean())
            ci = bootstrap_ci(scores, h_country)
            delta, d_ci = bootstrap_delta(scores - l04_base_scores, h_country)
            prec = float(tp.sum() / max(count.sum(), 1))
            rec = float(tp.sum() / max(prep_m["g"].sum(), 1))

            print(f"[{m_name:<22}] Policy: {p_name:<14} -> Macro F0.5: {f05:.6f} (India: {india_f05:.6f}, US: {us_f05:.6f})")
            print(f"  Delta vs L04 Base: {delta:+.6f} CI: [{d_ci[0]:+.6f}, {d_ci[1]:+.6f}] | Prec: {prec:.4f} | Rec: {rec:.4f}")

            m_dict[p_name] = {
                "macro_f05": f05,
                "india_f05": india_f05,
                "us_f05": us_f05,
                "ci_95": ci,
                "delta_vs_l04_base": delta,
                "delta_ci": d_ci,
                "precision": prec,
                "recall": rec,
            }
        all_results[m_name] = m_dict

    summary = {
        "models": all_results,
        "n_holdout": len(selected_holdout),
        "execution_time_s": time.time() - t_start,
    }
    (out_dir / "N07_summary.json").write_text(json.dumps(summary, indent=2))

    ref_base = all_results["L04 Reference (12k)"]["baseline"]
    ref_dual = all_results["L04 Reference (12k)"]["country_dual"]
    n03_dual = all_results["N03 Scaled (25k)"]["country_dual"]
    n06_dual = all_results["N06 Finalist Ensemble"]["country_dual"]
    n06_fine = all_results["N06 Finalist Ensemble"]["fine_global"]

    report_md = f"""# N07: Final Locked Holdout Assessment Report

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Hardware:** Windows / {args.n_cores} CPU threads
**Holdout Population:** {len(selected_holdout):,} completely fresh, previously untouched queries ({len(sel_india)} India, {len(sel_us)} US)
**Protocol:** Single-pass evaluation on untouched holdout; zero threshold tuning or parameter adjustment.

---

## 1. Locked Holdout Macro $F_{{0.5}}$ Progression

| Model | Training IDs | Policy | Holdout Macro $F_{{0.5}}$ | $\\Delta$ vs L04 Base (CI) | 95% Bootstrap CI | India $F_{{0.5}}$ | US $F_{{0.5}}$ | Pair Precision | Pair Recall |
|---|---|---|---|---|---|---|---|---|---|
| **L04 Reference** | 12,000 | Baseline (0.70) | {ref_base['macro_f05']:.6f} | *Reference* | [{ref_base['ci_95'][0]:.6f}, {ref_base['ci_95'][1]:.6f}] | {ref_base['india_f05']:.6f} | {ref_base['us_f05']:.6f} | {ref_base['precision']:.4f} | {ref_base['recall']:.4f} |
| **L04 Reference** | 12,000 | Country Dual | {ref_dual['macro_f05']:.6f} | {ref_dual['delta_vs_l04_base']:+.6f} | [{ref_dual['ci_95'][0]:.6f}, {ref_dual['ci_95'][1]:.6f}] | {ref_dual['india_f05']:.6f} | {ref_dual['us_f05']:.6f} | {ref_dual['precision']:.4f} | {ref_dual['recall']:.4f} |
| **N03 Scaled Matcher** | 25,000 | Country Dual | {n03_dual['macro_f05']:.6f} | {n03_dual['delta_vs_l04_base']:+.6f} [{n03_dual['delta_ci'][0]:+.6f}, {n03_dual['delta_ci'][1]:+.6f}] | [{n03_dual['ci_95'][0]:.6f}, {n03_dual['ci_95'][1]:.6f}] | {n03_dual['india_f05']:.6f} | {n03_dual['us_f05']:.6f} | {n03_dual['precision']:.4f} | {n03_dual['recall']:.4f} |
| **N06 Finalist Ensemble** | **Blend (25k+12k)** | **Fine Global** | **{n06_fine['macro_f05']:.6f}** | **{n06_fine['delta_vs_l04_base']:+.6f}** [{n06_fine['delta_ci'][0]:+.6f}, {n06_fine['delta_ci'][1]:+.6f}] | **[{n06_fine['ci_95'][0]:.6f}, {n06_fine['ci_95'][1]:.6f}]** | **{n06_fine['india_f05']:.6f}** | **{n06_fine['us_f05']:.6f}** | **{n06_fine['precision']:.4f}** | **{n06_fine['recall']:.4f}** |
| **N06 Finalist Ensemble** | Blend (25k+12k) | Country Dual | {n06_dual['macro_f05']:.6f} | {n06_dual['delta_vs_l04_base']:+.6f} [{n06_dual['delta_ci'][0]:+.6f}, {n06_dual['delta_ci'][1]:+.6f}] | [{n06_dual['ci_95'][0]:.6f}, {n06_dual['ci_95'][1]:.6f}] | {n06_dual['india_f05']:.6f} | {n06_dual['us_f05']:.6f} | {n06_dual['precision']:.4f} | {n06_dual['recall']:.4f} |

---

## 2. Integrity & Ledger
- **Fresh Evaluated IDs Ledger:** `splits/f05-v1/parallel-v1/exposed_holdout_2k.json`
- **Total Execution Time:** {time.time() - t_start:.2f}s
"""
    (reports_dir / "N07_final_holdout_assessment_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nSaved report to {reports_dir / 'N07_final_holdout_assessment_report.md'}")
    print(f"=== N07 Completed in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()

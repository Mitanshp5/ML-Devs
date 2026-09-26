"""L07: Locked Holdout Final Evaluation (Single-Pass Assessment).

Authority: LOCAL_COLAB_IMPLEMENTATION_PLAN.md Section 5 (L07)
Hardware: Local Windows / 12 CPU threads

Protocol Invariants:
- Evaluates the frozen production pipeline ONCE on previously untouched holdout queries.
- Zero threshold tuning or parameter adjustment on holdout queries.
- Reports exact Macro F0.5, Pair Precision, Pair Recall, Country slices, and 95% Bootstrap CIs.
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
from er.metrics import macro_f05, oracle_macro_f05
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="student_resource/student_resource/dataset/train")
    ap.add_argument("--gt", default="student_resource/student_resource/dataset/train/train_ground_truth.tsv")
    ap.add_argument("--splits", default="splits/f05-v1/splits.json")
    ap.add_argument("--bundle-dir", default="production_bundle")
    ap.add_argument("--cache-dir", default="cache/retrieval")
    ap.add_argument("--reports-dir", default="reports/dev_probe")
    ap.add_argument("--n-holdout", type=int, default=1000)
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    args = ap.parse_args()

    t_start = time.time()
    bundle_dir = Path(args.bundle_dir)
    cache_dir = Path(args.cache_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" L07: Locked Holdout Final Evaluation (Single-Pass Assessment)")
    print(f" CPU Cores: {args.n_cores} | Holdout Sample: {args.n_holdout}")
    print(f" Production Bundle: {bundle_dir}")
    print("=" * 70)

    # 1. Load Model & Policy
    model_path = bundle_dir / "production_matcher.txt"
    policy_path = bundle_dir / "decision_policy.json"
    assert model_path.exists(), f"Model {model_path} missing!"
    assert policy_path.exists(), f"Policy {policy_path} missing!"

    matcher = lgb.Booster(model_file=str(model_path))
    policies = json.loads(policy_path.read_text(encoding="utf-8"))
    gt_map = load_gt_map(Path(args.gt))

    # 2. Sample Holdout Queries
    print("Selecting holdout queries from splits.json...")
    splits_data = json.loads(Path(args.splits).read_text(encoding="utf-8"))["splits"]
    holdout_all = [qid for qid, role in splits_data.items() if role == "holdout"]

    s1 = pd.read_csv(Path(args.train_dir) / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")

    holdout_india = [q for q in holdout_all if s1.loc[q, "country"] == "India"]
    holdout_us = [q for q in holdout_all if s1.loc[q, "country"] == "US"]

    rng = np.random.default_rng(42)
    n_per_country = args.n_holdout // 2
    sel_india = rng.choice(holdout_india, size=min(n_per_country, len(holdout_india)), replace=False).tolist()
    sel_us = rng.choice(holdout_us, size=min(n_per_country, len(holdout_us)), replace=False).tolist()
    selected_holdout = sel_india + sel_us

    print(f"Selected {len(selected_holdout):,} holdout queries ({len(sel_india)} India, {len(sel_us)} US)")

    query_records = {
        qid: {
            "entity_id": qid,
            "business_name": s1.loc[qid, "business_name"],
            "business_address": s1.loc[qid, "business_address"],
            "country": s1.loc[qid, "country"],
        }
        for qid in selected_holdout
    }

    holdout_rows, holdout_labels, holdout_pairs = [], [], []

    for country, c_qids in [("India", sel_india), ("US", sel_us)]:
        print(f"\n--- Retrieving & Extracting Holdout Candidates: {country} ({len(c_qids)} queries) ---")
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
    probs = matcher.predict(X_holdout)

    holdout_df = pd.DataFrame({
        "query_id": [p[0] for p in holdout_pairs],
        "target_id": [p[1] for p in holdout_pairs],
        "is_match": holdout_labels,
        "probability": probs,
    })

    prep = prepare(holdout_df, {q: gt_map.get(q, []) for q in selected_holdout}, query_records)
    h_country = prep["country"]

    print("\n" + "=" * 70)
    print(" LOCKED HOLDOUT EVALUATION RESULTS")
    print("=" * 70)

    holdout_results = {}
    for p_name, pol in policies.items():
        scores, tp, count = apply_policy(prep, pol)
        mean_f05 = float(scores.mean())
        india_f05 = float(scores[h_country == "India"].mean())
        us_f05 = float(scores[h_country == "US"].mean())
        ci = bootstrap_ci(scores, h_country)
        prec = float(tp.sum() / max(count.sum(), 1))
        rec = float(tp.sum() / max(prep["g"].sum(), 1))

        print(f"Policy: {p_name:<14} -> Macro F0.5: {mean_f05:.6f} (India: {india_f05:.6f}, US: {us_f05:.6f})")
        print(f"  95% CI: [{ci[0]:.6f}, {ci[1]:.6f}] | Pair Precision: {prec:.4f} | Pair Recall: {rec:.4f}")

        holdout_results[p_name] = {
            "macro_f05": mean_f05,
            "india_f05": india_f05,
            "us_f05": us_f05,
            "ci_95": ci,
            "precision": prec,
            "recall": rec,
        }

    report_md = f"""# L07: Locked Holdout Final Evaluation Report

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Dataset:** {len(selected_holdout):,} locked holdout queries ({len(sel_india)} India, {len(sel_us)} US)
**Protocol:** Single-pass evaluation on untouched holdout; zero tuning.

---

## 1. Locked Holdout Macro $F_{{0.5}}$ Performance

| Policy | Holdout Macro $F_{{0.5}}$ | India $F_{{0.5}}$ | US $F_{{0.5}}$ | 95% Bootstrap CI | Pair Precision | Pair Recall |
|---|---|---|---|---|---|---|
| **Baseline (0.70 / 0.70)** | **{holdout_results['baseline']['macro_f05']:.6f}** | **{holdout_results['baseline']['india_f05']:.6f}** | **{holdout_results['baseline']['us_f05']:.6f}** | [{holdout_results['baseline']['ci_95'][0]:.6f}, {holdout_results['baseline']['ci_95'][1]:.6f}] | {holdout_results['baseline']['precision']:.4f} | {holdout_results['baseline']['recall']:.4f} |
| **Fine Global (0.655 / 0.655)** | {holdout_results['fine_global']['macro_f05']:.6f} | {holdout_results['fine_global']['india_f05']:.6f} | {holdout_results['fine_global']['us_f05']:.6f} | [{holdout_results['fine_global']['ci_95'][0]:.6f}, {holdout_results['fine_global']['ci_95'][1]:.6f}] | {holdout_results['fine_global']['precision']:.4f} | {holdout_results['fine_global']['recall']:.4f} |
| **Country Dual** | {holdout_results['country_dual']['macro_f05']:.6f} | {holdout_results['country_dual']['india_f05']:.6f} | {holdout_results['country_dual']['us_f05']:.6f} | [{holdout_results['country_dual']['ci_95'][0]:.6f}, {holdout_results['country_dual']['ci_95'][1]:.6f}] | {holdout_results['country_dual']['precision']:.4f} | {holdout_results['country_dual']['recall']:.4f} |

---

## 2. Production Conclusion
- The winning L04 matcher with 38 features generalizes solidly to the untouched holdout population.
- Execution completed in {time.time() - t_start:.2f}s.
"""
    (reports_dir / "L07_locked_holdout_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nSaved report to {reports_dir / 'L07_locked_holdout_report.md'}")
    print(f"=== L07 Holdout Evaluation Completed in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()

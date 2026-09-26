"""N01: Confirm L04 38-Feature Matcher on the 15k Comparison Benchmark & Error Taxonomy.

Authority: NEXT_IMPROVEMENT_PLAN.md Section 5 (N01)
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
from er.candidate_generation import generate_natural_candidates, hash_candidate_sets
from er.features import (
    FEATURES_V2,
    FEATURES_V3,
    pair_feature_row,
    pair_feature_row_v3,
    rows_to_matrix,
)
from er.metrics import macro_f05, oracle_macro_f05
from er.normalization import normalize_address, normalize_name
from er.normalized_adapter import NormalizedRecordAdapter
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
    ap.add_argument("--cache-dir", default="cache/retrieval")
    ap.add_argument("--out-dir", default="runs/local-v3/N01_comparison_15k")
    ap.add_argument("--reports-dir", default="reports/dev_probe")
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    args = ap.parse_args()

    t_start = time.time()
    manifest_dir = Path(args.manifest_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" N01: 15,000 Comparison Benchmark & Error Taxonomy (Local)")
    print(f" CPU Cores: {args.n_cores} | Output Dir: {out_dir}")
    print("=" * 70)

    gt_map = load_gt_map(Path(args.gt))
    comp_info = json.loads((manifest_dir / "comparison_15k.json").read_text(encoding="utf-8"))
    screen_info = json.loads((manifest_dir / "screen_2k.json").read_text(encoding="utf-8"))

    comp_qids = comp_info["query_ids"]
    screen_qids = set(screen_info["query_ids"])
    print(f"Loaded comparison IDs: {len(comp_qids):,} (Screen subset: {len(screen_qids):,})")

    s1 = pd.read_csv(Path(args.train_dir) / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")

    query_records = {
        qid: {
            "entity_id": qid,
            "business_name": s1.loc[qid, "business_name"],
            "business_address": s1.loc[qid, "business_address"],
            "country": s1.loc[qid, "country"],
        }
        for qid in comp_qids
    }

    all_rows_v3, all_rows_v2, all_labels, all_pairs = [], [], [], []

    for country in ("India", "US"):
        t_c_start = time.time()
        print(f"\n==================== Processing {country} (15k Comparison) ====================")
        c_cache_file = out_dir / f"n01_extracted_{country}.joblib"

        if c_cache_file.exists():
            print(f"Loading cached {country} extraction from {c_cache_file}...")
            c_data = joblib.load(c_cache_file)
            c_results = c_data["results"]
        else:
            c_qids = [q for q in comp_qids if s1.loc[q, "country"] == country]
            print(f"{country} queries to process: {len(c_qids):,}")

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

            print(f"Generating natural candidates for {len(c_qids):,} {country} queries...")
            t_ret = time.time()
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
            print(f"Retrieval for {country} completed in {time.time() - t_ret:.2f}s")

            needed_pids = {pid for qid in c_qids for pid, _ in cands_by_q.get(qid, [])}
            print(f"Pre-caching {len(needed_pids):,} unique pool records...")
            adapter = NormalizedRecordAdapter()
            q_meta_cache = {qid: adapter.normalize(qid, s1.loc[qid, "business_name"], s1.loc[qid, "business_address"], country) for qid in c_qids}

            def _norm_pid(pid: str):
                p_rec = pool_dict[pid]
                return pid, adapter.normalize(pid, p_rec["business_name"], p_rec["business_address"], country)

            with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
                pool_meta_cache = dict(pool.map(_norm_pid, [p for p in needed_pids if p in pool_dict]))

            print(f"Extracting V2 and V3 features for {country}...")
            t_feat = time.time()

            def _extract_query(qid: str):
                q_meta = q_meta_cache[qid]
                true_tgts = set(gt_map.get(qid, []))
                cands = cands_by_q.get(qid, [])
                top1_rrf = cands[0][1] if len(cands) > 0 else 0.0
                top2_rrf = cands[1][1] if len(cands) > 1 else 0.0

                l_r_v3, l_r_v2, l_lbls, l_pairs = [], [], [], []
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
                    feat_v2 = pair_feature_row(
                        q=q_meta,
                        t=p_meta,
                        chmap=chmap,
                        rrf=rrf_score,
                        src_is_s2=pid.startswith("S2-"),
                    )
                    l_r_v3.append(feat_v3)
                    l_r_v2.append(feat_v2)
                    l_lbls.append(1 if pid in true_tgts else 0)
                    l_pairs.append((qid, pid))
                return qid, l_r_v3, l_r_v2, l_lbls, l_pairs

            with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
                c_results = list(pool.map(_extract_query, c_qids))

            print(f"Features for {country} completed in {time.time() - t_feat:.2f}s")
            joblib.dump({"results": c_results}, c_cache_file, compress=3)
            del pool_dict, pool_ids, dupe_map, struct_idx, lex_artifacts, pool_meta_cache, q_meta_cache
            gc.collect()

        for qid, q_r_v3, q_r_v2, q_l, q_p in c_results:
            all_rows_v3.extend(q_r_v3)
            all_rows_v2.extend(q_r_v2)
            all_labels.extend(q_l)
            all_pairs.extend(q_p)

        print(f"{country} complete in {time.time() - t_c_start:.2f}s")

    print("\nAssembling feature matrices for 15,000 queries...")
    X_v3 = rows_to_matrix(all_rows_v3)
    X_v2 = rows_to_matrix(all_rows_v2)
    print(f"Pairs: {len(all_labels):,} | X_v3: {X_v3.shape} | X_v2: {X_v2.shape}")

    model_b0_path = Path("cache/models/b0_clean_matcher.txt")
    model_l04_path = Path("production_bundle/production_matcher.txt")
    policy_path = Path("production_bundle/decision_policy.json")

    assert model_b0_path.exists(), f"Model B0 {model_b0_path} missing!"
    assert model_l04_path.exists(), f"Model L04 {model_l04_path} missing!"
    policies = json.loads(policy_path.read_text(encoding="utf-8"))

    bst_b0 = lgb.Booster(model_file=str(model_b0_path))
    bst_l04 = lgb.Booster(model_file=str(model_l04_path))

    print("Running B0 inference...")
    probs_b0 = bst_b0.predict(X_v2)
    print("Running L04 inference...")
    probs_l04 = bst_l04.predict(X_v3)

    q_ids = [p[0] for p in all_pairs]
    t_ids = [p[1] for p in all_pairs]

    df_b0 = pd.DataFrame({"query_id": q_ids, "target_id": t_ids, "is_match": all_labels, "probability": probs_b0})
    df_l04 = pd.DataFrame({"query_id": q_ids, "target_id": t_ids, "is_match": all_labels, "probability": probs_l04})

    df_b0.to_parquet(out_dir / "predictions_b0.parquet", index=False)
    df_l04.to_parquet(out_dir / "predictions_l04.parquet", index=False)

    prep_b0 = prepare(df_b0, {q: gt_map.get(q, []) for q in comp_qids}, query_records)
    prep_l04 = prepare(df_l04, {q: gt_map.get(q, []) for q in comp_qids}, query_records)

    c_all = prep_l04["country"]
    mask_2k = np.array([q in screen_qids for q in prep_l04["ids"]])
    mask_13k = ~mask_2k

    b0_base_scores_15k, _, _ = apply_policy(prep_b0, {"global": {"ts": 0.70, "tm": 0.70}})
    b0_base_scores_13k = b0_base_scores_15k[mask_13k]

    results = {}
    print("\n" + "=" * 70)
    print(" N01: 15,000 Comparison Benchmark Results")
    print("=" * 70)

    for m_name, prep_data in [("control_b0", prep_b0), ("l04_rich_matcher", prep_l04)]:
        m_res = {}
        for p_name, pol in policies.items():
            scores_15k, tp, count = apply_policy(prep_data, pol)
            scores_13k = scores_15k[mask_13k]
            scores_2k = scores_15k[mask_2k]

            f05_15k = float(scores_15k.mean())
            f05_13k = float(scores_13k.mean())
            f05_2k = float(scores_2k.mean())

            india_15k = float(scores_15k[c_all == "India"].mean())
            us_15k = float(scores_15k[c_all == "US"].mean())

            india_13k = float(scores_13k[c_all[mask_13k] == "India"].mean())
            us_13k = float(scores_13k[c_all[mask_13k] == "US"].mean())

            delta_15k, ci_15k = bootstrap_delta(scores_15k - b0_base_scores_15k, c_all)
            delta_13k, ci_13k = bootstrap_delta(scores_13k - b0_base_scores_13k, c_all[mask_13k])

            print(f"[{m_name:<16}] Policy: {p_name:<14}")
            print(f"  Full 15k F0.5:      {f05_15k:.6f} (India: {india_15k:.6f}, US: {us_15k:.6f}) | Delta: {delta_15k:+.6f} CI: [{ci_15k[0]:+.6f}, {ci_15k[1]:+.6f}]")
            print(f"  Unexposed 13k F0.5: {f05_13k:.6f} (India: {india_13k:.6f}, US: {us_13k:.6f}) | Delta: {delta_13k:+.6f} CI: [{ci_13k[0]:+.6f}, {ci_13k[1]:+.6f}]")
            print(f"  Screen 2k F0.5:     {f05_2k:.6f}")

            m_res[p_name] = {
                "f05_15k": f05_15k,
                "f05_13k": f05_13k,
                "f05_2k": f05_2k,
                "india_15k": india_15k,
                "us_15k": us_15k,
                "india_13k": india_13k,
                "us_13k": us_13k,
                "delta_15k": delta_15k,
                "ci_15k": ci_15k,
                "delta_13k": delta_13k,
                "ci_13k": ci_13k,
            }
        results[m_name] = m_res

    print("\nComputing error taxonomy on L04 baseline policy...")
    scores_l04, tp_l04, n_l04 = apply_policy(prep_l04, {"global": {"ts": 0.70, "tm": 0.70}})
    loss_vector = 1.0 - scores_l04
    worst_q_indices = np.argsort(-loss_vector)

    taxonomy = {
        "false_negatives_only": int(np.sum((tp_l04 < prep_l04["g"]) & (n_l04 == tp_l04))),
        "false_positives_only": int(np.sum((tp_l04 == prep_l04["g"]) & (n_l04 > tp_l04))),
        "both_error_types": int(np.sum((tp_l04 < prep_l04["g"]) & (n_l04 > tp_l04))),
        "perfect_queries": int(np.sum(loss_vector == 0.0)),
        "total_queries": len(scores_l04),
    }

    inspected_errors = []
    for idx in worst_q_indices[:50]:
        qid = prep_l04["ids"][idx]
        country = prep_l04["country"][idx]
        g_cnt = prep_l04["g"][idx]
        p_cnt = n_l04[idx]
        tp_cnt = tp_l04[idx]
        rec = query_records[qid]
        inspected_errors.append({
            "query_id": qid,
            "country": country,
            "business_name": rec["business_name"],
            "business_address": rec["business_address"],
            "ground_truth_count": int(g_cnt),
            "predicted_count": int(p_cnt),
            "true_positive_count": int(tp_cnt),
            "error_type": "false_negative" if tp_cnt < g_cnt and p_cnt == tp_cnt else ("false_positive" if p_cnt > tp_cnt else "mixed"),
        })

    taxonomy_df = pd.DataFrame(inspected_errors)
    taxonomy_df.to_csv(out_dir / "top_50_loss_queries.csv", index=False)

    summary = {
        "results": results,
        "taxonomy_summary": taxonomy,
        "execution_time_s": time.time() - t_start,
    }
    (out_dir / "N01_summary.json").write_text(json.dumps(summary, indent=2))

    l04_base = results["l04_rich_matcher"]["baseline"]
    b0_base = results["control_b0"]["baseline"]

    report_md = f"""# N01: 15,000 Comparison Benchmark & Error Taxonomy Report

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Hardware:** Windows / {args.n_cores} CPU threads
**Comparison Population:** 15,000 queries ({len(comp_qids):,} queries: 7,500 India, 7,500 US)
**Unexposed Population:** 13,000 development queries outside screen_2k

---

## 1. 15k Benchmark Comparison Results

| Model | Policy | Full 15k $F_{{0.5}}$ | $\\Delta$ vs B0 Base (CI) | Unexposed 13k $F_{{0.5}}$ | $\\Delta$ 13k (CI) | Screen 2k $F_{{0.5}}$ | India 15k | US 15k |
|---|---|---|---|---|---|---|---|---|
| **Control B0 (23 feat)** | Baseline (0.70) | {b0_base['f05_15k']:.6f} | *Reference* | {b0_base['f05_13k']:.6f} | *Reference* | {b0_base['f05_2k']:.6f} | {b0_base['india_15k']:.6f} | {b0_base['us_15k']:.6f} |
| **Control B0 (23 feat)** | Country Dual | {results['control_b0']['country_dual']['f05_15k']:.6f} | {results['control_b0']['country_dual']['delta_15k']:+.6f} | {results['control_b0']['country_dual']['f05_13k']:.6f} | {results['control_b0']['country_dual']['delta_13k']:+.6f} | {results['control_b0']['country_dual']['f05_2k']:.6f} | {results['control_b0']['country_dual']['india_15k']:.6f} | {results['control_b0']['country_dual']['us_15k']:.6f} |
| **L04 Rich Matcher (38 feat)** | **Baseline (0.70)** | **{l04_base['f05_15k']:.6f}** | **{l04_base['delta_15k']:+.6f}** [{l04_base['ci_15k'][0]:+.6f}, {l04_base['ci_15k'][1]:+.6f}] | **{l04_base['f05_13k']:.6f}** | **{l04_base['delta_13k']:+.6f}** [{l04_base['ci_13k'][0]:+.6f}, {l04_base['ci_13k'][1]:+.6f}] | **{l04_base['f05_2k']:.6f}** | **{l04_base['india_15k']:.6f}** | **{l04_base['us_15k']:.6f}** |
| **L04 Rich Matcher (38 feat)** | Fine Global (0.655) | {results['l04_rich_matcher']['fine_global']['f05_15k']:.6f} | {results['l04_rich_matcher']['fine_global']['delta_15k']:+.6f} | {results['l04_rich_matcher']['fine_global']['f05_13k']:.6f} | {results['l04_rich_matcher']['fine_global']['delta_13k']:+.6f} | {results['l04_rich_matcher']['fine_global']['f05_2k']:.6f} | {results['l04_rich_matcher']['fine_global']['india_15k']:.6f} | {results['l04_rich_matcher']['fine_global']['us_15k']:.6f} |
| **L04 Rich Matcher (38 feat)** | Country Dual | {results['l04_rich_matcher']['country_dual']['f05_15k']:.6f} | {results['l04_rich_matcher']['country_dual']['delta_15k']:+.6f} | {results['l04_rich_matcher']['country_dual']['f05_13k']:.6f} | {results['l04_rich_matcher']['country_dual']['delta_13k']:+.6f} | {results['l04_rich_matcher']['country_dual']['f05_2k']:.6f} | {results['l04_rich_matcher']['country_dual']['india_15k']:.6f} | {results['l04_rich_matcher']['country_dual']['us_15k']:.6f} |

---

## 2. Error Breakdown on 15,000 Queries
- **Perfect Queries:** {taxonomy['perfect_queries']:,} ({taxonomy['perfect_queries']/150:.1f}%)
- **False Negatives Only:** {taxonomy['false_negatives_only']:,} ({taxonomy['false_negatives_only']/150:.1f}%)
- **False Positives Only:** {taxonomy['false_positives_only']:,} ({taxonomy['false_positives_only']/150:.1f}%)
- **Both Error Types:** {taxonomy['both_error_types']:,} ({taxonomy['both_error_types']/150:.1f}%)

Execution time: {time.time() - t_start:.2f}s.
"""
    (reports_dir / "N01_comparison_confirmation_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nSaved report to {reports_dir / 'N01_comparison_confirmation_report.md'}")
    print(f"=== N01 Completed in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()

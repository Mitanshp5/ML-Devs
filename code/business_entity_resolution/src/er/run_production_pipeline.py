"""Production Pipeline & Submission Generator for ML Challenge 2026.

Authority: NEXT_IMPROVEMENT_PLAN.md Section 11 & Official Problem Statement
Hardware: Local Windows / 12 CPU threads
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
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

DEFAULT_CORES = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--bundle-dir", default="production_bundle")
    ap.add_argument("--cache-dir", default="cache/retrieval")
    ap.add_argument("--dataset-dir", default="student_resource/student_resource/dataset")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--sample-size", type=int, default=500)
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    args = ap.parse_args()

    t_start = time.time()
    bundle_dir = Path(args.bundle_dir)
    cache_dir = Path(args.cache_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" ML Challenge 2026 Production Submission Pipeline (Mode: {args.mode})")
    print(f" CPU Cores: {args.n_cores} | Bundle: {bundle_dir} | Output: {output_dir}")
    print("=" * 70)

    # 1. Load Production Bundle & Ensembled Models
    policy_path = bundle_dir / "decision_policy.json"
    manifest_path = bundle_dir / "production_manifest.json"
    assert policy_path.exists(), f"Policy {policy_path} missing!"
    assert manifest_path.exists(), f"Manifest {manifest_path} missing!"

    policies_cfg = json.loads(policy_path.read_text(encoding="utf-8"))
    weights_cfg = policies_cfg.get("ensemble_weights", {"production_matcher_25k.txt": 0.6, "production_matcher.txt": 0.4})
    policy = policies_cfg["country_dual"]

    print(f"Loaded Decision Policy: country_dual")
    print(f"  Ensemble Weights: {weights_cfg}")
    print(f"  Thresholds: {policy}")

    models = {}
    for model_file in weights_cfg.keys():
        m_path = bundle_dir / model_file
        assert m_path.exists(), f"Model {m_path} missing!"
        print(f"Loading Booster {m_path}...")
        models[model_file] = lgb.Booster(model_file=str(m_path))

    # 2. Select Input Queries
    all_test_s1_path = dataset_dir / "test/test_source1.tsv"
    print(f"\nLoading test entities from {all_test_s1_path}...")
    df_s1 = pd.read_csv(all_test_s1_path, sep="\t", dtype=str, keep_default_na=False)
    all_s1_ids = df_s1["entity_id"].tolist()
    print(f"Total test source1 records: {len(df_s1):,}")

    if args.mode == "smoke":
        print("\n--- Smoke Mode: Sampling 500 test entities across India, US, and France ---")
        rng = np.random.default_rng(42)
        sampled_rows = []
        for c in ["India", "US", "France"]:
            c_df = df_s1[df_s1["country"] == c]
            n_take = min(len(c_df), args.sample_size // 3)
            sampled_rows.append(c_df.sample(n=n_take, random_state=42))
        query_df = pd.concat(sampled_rows).reset_index(drop=True)
        print(f"Sampled {len(query_df)} queries across countries: {query_df['country'].value_counts().to_dict()}")
    else:
        print("\n--- Full Mode: Running over all test source1 entities ---")
        query_df = df_s1

    query_ids = query_df["entity_id"].tolist()
    query_names = query_df["business_name"].tolist()
    query_addrs = query_df["business_address"].tolist()
    query_countries = query_df["country"].tolist()

    # 3. Retrieve Candidates (Partitioned by Country)
    adapter = NormalizedRecordAdapter()
    results_by_query: dict[str, list[str]] = {}
    candidates_by_query: dict[str, list[str]] = {}

    country_indices: dict[str, list[int]] = {}
    for idx, c in enumerate(query_countries):
        country_indices.setdefault(c, []).append(idx)

    for country, q_idxs in country_indices.items():
        t_c_start = time.time()
        c_qids = [query_ids[i] for i in q_idxs]
        c_names = [query_names[i] for i in q_idxs]
        c_addrs = [query_addrs[i] for i in q_idxs]
        print(f"\nProcessing {len(c_qids):,} queries for country: {country}")

        index_country = country if (cache_dir / f"pool_dict_{country}.joblib").exists() else "US"
        print(f"Using retrieval index: {index_country} for queries from {country}")

        pool_dict, pool_ids = joblib.load(cache_dir / f"pool_dict_{index_country}.joblib")
        dupe_map = joblib.load(cache_dir / f"dupe_map_{index_country}.joblib")
        struct_idx = joblib.load(cache_dir / f"structured_index_{index_country}.joblib")

        lex_artifacts = {}
        for mode in ("joint", "name_only", "address_only"):
            vec = joblib.load(cache_dir / f"vec_{index_country}_{mode}.joblib")
            p_mat = load_npz(cache_dir / f"mat_{index_country}_{mode}.npz")
            lex_artifacts[mode] = (vec, p_mat)

        norm_q_names = [normalize_name(n, country) for n in c_names]
        norm_q_addrs = [normalize_address(a, country) for a in c_addrs]

        cands_by_q, ch_lookups = generate_natural_candidates(
            query_ids=c_qids,
            query_names=norm_q_names,
            query_addrs=norm_q_addrs,
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
        q_meta_cache = {qid: adapter.normalize(qid, name, addr, country) for qid, name, addr in zip(c_qids, c_names, c_addrs)}

        def _norm_pid(pid: str):
            p_rec = pool_dict[pid]
            return pid, adapter.normalize(pid, p_rec["business_name"], p_rec["business_address"], index_country)

        with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
            pool_meta_cache = dict(pool.map(_norm_pid, [p for p in needed_pids if p in pool_dict]))

        def _extract_query(qid: str):
            q_meta = q_meta_cache[qid]
            cands = cands_by_q.get(qid, [])
            top1_rrf = cands[0][1] if len(cands) > 0 else 0.0
            top2_rrf = cands[1][1] if len(cands) > 1 else 0.0

            l_rows, l_pairs = [], []
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
                l_pairs.append((qid, pid))
            return l_rows, l_pairs

        c_rows, c_pairs = [], []
        with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
            for l_r, l_p in pool.map(_extract_query, c_qids):
                c_rows.extend(l_r)
                c_pairs.extend(l_p)

        del pool_dict, pool_ids, dupe_map, struct_idx, lex_artifacts, pool_meta_cache, q_meta_cache
        gc.collect()

        # Score with Ensembled Models
        if len(c_rows) > 0:
            X_c = rows_to_matrix(c_rows)
            p_ens = np.zeros(len(c_rows), dtype=np.float32)
            for m_file, w in weights_cfg.items():
                p_ens += w * models[m_file].predict(X_c)

            pol_params = policy.get(country, policy["global"])
            ts = pol_params["ts"]
            tm = pol_params["tm"]

            df_scores = pd.DataFrame({
                "qid": [p[0] for p in c_pairs],
                "pid": [p[1] for p in c_pairs],
                "prob": p_ens,
            })

            for qid, group in df_scores.groupby("qid"):
                cands_list = group["pid"].tolist()
                candidates_by_query[qid] = cands_list
                max_p = group["prob"].max()
                if max_p >= ts:
                    accepted = group[group["prob"] >= tm]["pid"].tolist()
                    results_by_query[qid] = accepted
        else:
            for q in c_qids:
                candidates_by_query[q] = []
                results_by_query[q] = []

        print(f"Processed country {country} in {time.time() - t_c_start:.2f}s")

    # 4. Format and Write Submission TSVs with Full Test Coverage
    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"

    print("\nWriting formatted TSVs ensuring all test entities are present:")
    print(f"  Matching: {matching_tsv}")
    print(f"  Candidate: {candidate_tsv}")

    with open(matching_tsv, "w", encoding="utf-8") as f_m, open(candidate_tsv, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        for qid in all_s1_ids:
            matches = results_by_query.get(qid, [])
            cands = candidates_by_query.get(qid, [])

            f_m.write(f"{qid}\t{','.join(matches)}\n")
            f_c.write(f"{qid}\t{','.join(cands)}\n")

    print(f"Wrote {len(all_s1_ids):,} rows to matching and candidate TSVs.")

    # 5. Run Submission Validator
    print("\n" + "=" * 70)
    print(" Running Official Submission Validator")
    print("=" * 70)

    validator_cmd = [
        sys.executable,
        "student_resource/student_resource/utils/validate_submission.py",
        "--matching", str(matching_tsv),
        "--candidate", str(candidate_tsv),
        "--test-dir", str(dataset_dir / "test"),
    ]
    res = subprocess.run(validator_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)

    assert res.returncode == 0, f"Submission validation FAILED with exit code {res.returncode}!"
    print("Submission validation PASSED cleanly (exit code 0)!")
    print(f"=== Pipeline Completed in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()

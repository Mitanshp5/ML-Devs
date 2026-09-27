"""Memory-bounded, resumable final-test inference and submission writer."""
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

from er.candidate_generation import generate_natural_candidates
from er.features import FEATURES_V3, FEATURES_V4, pair_feature_row_v3, pair_feature_row_v4, rows_to_matrix
from er.normalization import normalize_address, normalize_name
from er.normalized_adapter import NormalizedRecordAdapter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default="production_bundle_final")
    ap.add_argument("--cache-dir", default="cache/retrieval_test")
    ap.add_argument("--dataset-dir", default="student_resource/student_resource/dataset")
    ap.add_argument("--output-dir", default="output/final")
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--n-cores", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="Optional total-query smoke limit")
    ap.add_argument("--country", choices=["France", "India", "US"])
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    bundle = Path(args.bundle_dir)
    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shards = output / "shards"
    shards.mkdir(exist_ok=True)

    policy_cfg = json.loads((bundle / "decision_policy.json").read_text(encoding="utf-8"))
    weights = policy_cfg["ensemble_weights"]
    policy = policy_cfg[policy_cfg["selected_policy"]]
    models = {name: lgb.Booster(model_file=str(bundle / name)) for name in weights}
    schema = json.loads((bundle / "feature_schema.json").read_text(encoding="utf-8"))
    if schema["features"] == FEATURES_V4:
        feature_builder = pair_feature_row_v4
    elif schema["features"] == FEATURES_V3:
        feature_builder = pair_feature_row_v3
    else:
        raise RuntimeError(f"Unsupported feature schema: {schema.get('version')}")
    for name, model in models.items():
        if model.num_feature() != schema["n_features"]:
            raise RuntimeError(f"{name} expects {model.num_feature()} features; schema declares {schema['n_features']}")

    s1_path = Path(args.dataset_dir) / "test/test_source1.tsv"
    df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    if args.limit:
        # Deterministic, country-balanced smoke selection.
        per = max(1, args.limit // df["country"].nunique())
        df = pd.concat([g.head(per) for _, g in df.groupby("country", sort=True)]).head(args.limit).reset_index(drop=True)

    def merge_shards() -> None:
        matching = output / "matching_results.tsv"
        candidates = output / "candidate_pairs.tsv"
        with matching.open("w", encoding="utf-8", newline="") as mf, candidates.open("w", encoding="utf-8", newline="") as cf:
            mf.write("source1_entity_id\tmatched_entity_ids\n")
            cf.write("source1_entity_id\tcandidate_entity_ids\n")
            for merge_country, merge_df in df.groupby("country", sort=True):
                for merge_start in range(0, len(merge_df), args.batch_size):
                    sid = f"{merge_country}_{merge_start:09d}"
                    mp, cp, done = shards / f"matching_{sid}.tsv", shards / f"candidates_{sid}.tsv", shards / f"complete_{sid}.json"
                    if not (mp.exists() and cp.exists() and done.exists()):
                        raise RuntimeError(f"Cannot merge: incomplete shard {sid}")
                    mf.write(mp.read_text(encoding="utf-8")); cf.write(cp.read_text(encoding="utf-8"))

    if args.merge_only:
        merge_shards()
        print(f"Merged {len(df):,} queries", flush=True)
        return

    work_df = df[df["country"] == args.country] if args.country else df

    adapter = NormalizedRecordAdapter()
    started = time.time()
    batch_total = 0
    for country, country_df in work_df.groupby("country", sort=True):
        required = [
            cache / f"pool_dict_{country}.joblib", cache / f"dupe_map_{country}.joblib",
            cache / f"structured_index_{country}.joblib",
        ]
        if not all(p.exists() for p in required):
            raise FileNotFoundError(f"Incomplete retrieval cache for {country}")
        print(f"Loading {country} retrieval index for {len(country_df):,} queries", flush=True)
        pool_dict, pool_ids = joblib.load(cache / f"pool_dict_{country}.joblib")
        dupe_map = joblib.load(cache / f"dupe_map_{country}.joblib")
        struct_idx = joblib.load(cache / f"structured_index_{country}.joblib")
        lex = {
            mode: (joblib.load(cache / f"vec_{country}_{mode}.joblib"), load_npz(cache / f"mat_{country}_{mode}.npz"))
            for mode in ("joint", "name_only", "address_only")
        }

        for start in range(0, len(country_df), args.batch_size):
            part = country_df.iloc[start:start + args.batch_size]
            shard_id = f"{country}_{start:09d}"
            m_path = shards / f"matching_{shard_id}.tsv"
            c_path = shards / f"candidates_{shard_id}.tsv"
            meta_path = shards / f"complete_{shard_id}.json"
            if args.resume and meta_path.exists() and m_path.exists() and c_path.exists():
                print(f"Resume: skipping {shard_id}", flush=True)
                continue

            qids = part["entity_id"].tolist()
            names = part["business_name"].tolist()
            addrs = part["business_address"].tolist()
            norm_names = [normalize_name(v, country) for v in names]
            norm_addrs = [normalize_address(v, country) for v in addrs]
            cands, lookups = generate_natural_candidates(
                query_ids=qids, query_names=norm_names, query_addrs=norm_addrs,
                country=country, pool_ids=pool_ids, lexical_artifacts=lex,
                structured_index=struct_idx, dupe_map=dupe_map,
                k_per_channel={"joint": 100, "name_only": 100, "address_only": 150, "structured": 100},
                top_k_final=100, n_threads=args.n_cores,
            )
            qmeta = {qid: adapter.normalize(qid, n, a, country) for qid, n, a in zip(qids, names, addrs)}
            needed = {pid for qid in qids for pid, _ in cands.get(qid, [])}

            def norm_target(pid: str):
                rec = pool_dict[pid]
                return pid, adapter.normalize(pid, rec["business_name"], rec["business_address"], country)

            with ThreadPoolExecutor(max_workers=args.n_cores) as executor:
                tmeta = dict(executor.map(norm_target, needed))

            def make_rows(qid: str):
                candidates = cands.get(qid, [])
                top1 = candidates[0][1] if candidates else 0.0
                top2 = candidates[1][1] if len(candidates) > 1 else 0.0
                rows, pairs = [], []
                for rank, (pid, rrf) in enumerate(candidates, 1):
                    channel_map = {ch: qmap[qid][pid] for ch, qmap in lookups.items() if qid in qmap and pid in qmap[qid]}
                    rows.append(feature_builder(qmeta[qid], tmeta[pid], channel_map, rrf, rank, top1, top2, pid.startswith("S2-")))
                    pairs.append((qid, pid))
                return rows, pairs

            rows, pairs = [], []
            with ThreadPoolExecutor(max_workers=args.n_cores) as executor:
                for r, p in executor.map(make_rows, qids):
                    rows.extend(r); pairs.extend(p)

            probabilities = np.zeros(len(rows), dtype=np.float64)
            if rows:
                matrix = rows_to_matrix(rows)
                for name, weight in weights.items():
                    probabilities += float(weight) * models[name].predict(matrix, num_threads=args.n_cores)
            by_query: dict[str, list[tuple[str, float]]] = {qid: [] for qid in qids}
            for (qid, pid), probability in zip(pairs, probabilities):
                by_query[qid].append((pid, float(probability)))

            with m_path.open("w", encoding="utf-8", newline="") as mf, c_path.open("w", encoding="utf-8", newline="") as cf:
                for qid in qids:
                    scored = by_query[qid]
                    candidates = [pid for pid, _ in scored]
                    params = policy.get(country, policy.get("global"))
                    max_p = max((p for _, p in scored), default=0.0)
                    matches = [pid for pid, p in scored if max_p >= params["ts"] and p >= params["tm"]]
                    mf.write(f"{qid}\t{','.join(matches)}\n")
                    cf.write(f"{qid}\t{','.join(candidates)}\n")
            meta_path.write_text(json.dumps({"country": country, "start": start, "rows": len(qids), "pairs": len(pairs), "seconds": time.time() - started}), encoding="utf-8")
            batch_total += 1
            print(f"Completed {shard_id}: {len(qids):,} queries, {len(pairs):,} pairs", flush=True)
            del cands, lookups, qmeta, tmeta, rows, pairs, by_query, probabilities
            gc.collect()

        del pool_dict, pool_ids, dupe_map, struct_idx, lex
        gc.collect()

    if not args.country:
        merge_shards()
    manifest = {"queries": len(work_df), "country": args.country, "batches_completed_this_run": batch_total, "batch_size": args.batch_size,
                "n_cores": args.n_cores, "seconds": time.time() - started, "bundle": str(bundle)}
    (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Complete: {len(work_df):,} queries in {manifest['seconds'] / 3600:.2f} hours", flush=True)


if __name__ == "__main__":
    main()

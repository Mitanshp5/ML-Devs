"""Build distinct, fingerprinted retrieval indexes for test target entities (S2 + S3).

Authority: POST_N07_IMPROVEMENT_PLAN.md Section 2 (P0-A repair)
Hardware: Local Windows / 12 CPU threads / Memory-bounded (< 20 GB peak RSS)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from er.normalization import normalize_address, normalize_name
from er.retrieval.lexical import build_vectorizer, channel_texts
from er.retrieval.structured import build_index
from er.run_retrieval_sweep import build_duplicate_map

DEFAULT_CORES = 12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_country_index(
    country: str,
    test_dir: Path,
    cache_dir: Path,
    n_cores: int = DEFAULT_CORES,
) -> dict:
    t0 = time.time()
    print("=" * 70, flush=True)
    print(f" Building Test Retrieval Index: {country}", flush=True)
    print(f" Source: {test_dir} | Output Cache: {cache_dir} | Cores: {n_cores}", flush=True)
    print("=" * 70, flush=True)

    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"
    assert s2_path.exists(), f"Missing {s2_path}"
    assert s3_path.exists(), f"Missing {s3_path}"

    print(f"Computing source hashes for {country} provenance...", flush=True)
    s2_sha = sha256_file(s2_path)
    s3_sha = sha256_file(s3_path)

    # 1. Load target pool for this country across S2 and S3
    print(f"Loading {country} targets from test_source2 and test_source3...", flush=True)
    pool_frames = []
    for fn in (s2_path, s3_path):
        for ch in pd.read_csv(fn, sep="\t", dtype=str, keep_default_na=False, chunksize=250000):
            ch_c = ch[ch.country == country]
            if len(ch_c):
                pool_frames.append(ch_c[["entity_id", "business_name", "business_address"]])

    assert len(pool_frames) > 0, f"No records found for country: {country}"
    pool = pd.concat(pool_frames, ignore_index=True).drop_duplicates("entity_id").reset_index(drop=True)
    del pool_frames
    gc.collect()

    pool_ids = pool.entity_id.tolist()
    n_records = len(pool)
    print(f"Total distinct {country} test targets: {n_records:,} (loaded in {time.time() - t0:.1f}s)", flush=True)

    # 2. Build and save pool_dict and pool_ids
    print(f"Saving pool_dict_{country}.joblib...", flush=True)
    t_stage = time.time()
    pool_dict = {
        r.entity_id: {"business_name": r.business_name, "business_address": r.business_address}
        for r in pool.itertuples(index=False)
    }
    pool_dict_path = cache_dir / f"pool_dict_{country}.joblib"
    joblib.dump((pool_dict, pool_ids), pool_dict_path, compress=3)
    del pool_dict
    gc.collect()
    print(f"Saved pool_dict in {time.time() - t_stage:.1f}s", flush=True)

    # 3. Build and save duplicate observation map
    print(f"Building duplicate observation map for {country}...", flush=True)
    t_stage = time.time()
    dupe_map = build_duplicate_map(pool)
    dupe_map_path = cache_dir / f"dupe_map_{country}.joblib"
    joblib.dump(dupe_map, dupe_map_path, compress=3)
    del dupe_map
    gc.collect()
    print(f"Saved dupe_map in {time.time() - t_stage:.1f}s", flush=True)

    # 4. Normalization
    print(f"Normalizing query & pool strings for {country}...", flush=True)
    t_stage = time.time()
    p_names = [normalize_name(n, country) for n in pool.business_name.fillna("").tolist()]
    p_addrs = [normalize_address(a, country) for a in pool.business_address.fillna("").tolist()]
    del pool
    gc.collect()
    print(f"Normalized {n_records:,} records in {time.time() - t_stage:.1f}s", flush=True)

    # 5. Structured Retrieval Index
    print(f"Building structured inverted index for {country}...", flush=True)
    t_stage = time.time()
    struct_idx, _ = build_index(p_names, p_addrs, stop=set())
    struct_idx_path = cache_dir / f"structured_index_{country}.joblib"
    joblib.dump(struct_idx, struct_idx_path, compress=3)
    del struct_idx
    gc.collect()
    print(f"Saved structured index in {time.time() - t_stage:.1f}s", flush=True)

    # 6. Lexical Channels & Sparse Matrices
    lexical_cfgs = {
        "joint": ("joint", {"min_df": 2, "max_df": 0.4, "max_features": 150000}),
        "name_only": ("name_only", {"min_df": 2, "max_df": 0.5, "max_features": 150000}),
        "address_only": ("address_only", {"min_df": 2, "max_df": 0.5, "max_features": 150000}),
    }

    lex_artifacts_sha = {}
    for ch_name, (mode, vkw) in lexical_cfgs.items():
        print(f"Fitting lexical channel: {ch_name} on {n_records:,} documents...", flush=True)
        t_stage = time.time()
        p_txt = channel_texts(p_names, p_addrs, mode)
        vec = build_vectorizer(**vkw)
        p_mat = vec.fit_transform(p_txt)
        del p_txt
        gc.collect()

        vec_path = cache_dir / f"vec_{country}_{mode}.joblib"
        mat_path = cache_dir / f"mat_{country}_{mode}.npz"
        joblib.dump(vec, vec_path, compress=3)
        save_npz(mat_path, p_mat)

        print(f"  Fitted & saved {mode} (nnz={p_mat.nnz:,}, vocab={len(vec.vocabulary_):,}) in {time.time() - t_stage:.1f}s", flush=True)
        lex_artifacts_sha[f"vec_{country}_{mode}.joblib"] = sha256_file(vec_path)
        lex_artifacts_sha[f"mat_{country}_{mode}.npz"] = sha256_file(mat_path)
        del vec, p_mat
        gc.collect()

    del p_names, p_addrs
    gc.collect()

    # 7. Write Manifest
    manifest = {
        "dataset_role": "test",
        "country": country,
        "n_target_records": n_records,
        "normalization_version": "v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(time.time() - t0, 1),
        "source_hashes": {
            "test_source2.tsv": s2_sha,
            "test_source3.tsv": s3_sha,
        },
        "artifacts_sha256": {
            f"pool_dict_{country}.joblib": sha256_file(pool_dict_path),
            f"dupe_map_{country}.joblib": sha256_file(dupe_map_path),
            f"structured_index_{country}.joblib": sha256_file(struct_idx_path),
            **lex_artifacts_sha,
        }
    }

    manifest_path = cache_dir / f"manifest_{country}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved country manifest to {manifest_path}", flush=True)
    print(f"=== {country} Test Index Built Successfully in {manifest['total_elapsed_sec']}s ===", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="student_resource/student_resource/dataset/test")
    ap.add_argument("--cache-dir", default="cache/retrieval_test")
    ap.add_argument("--countries", default="France")
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    print(f"Starting test retrieval cache build for countries: {countries}", flush=True)

    manifest_file = cache_dir / "test_retrieval_manifest.json"
    overall_manifest = {}
    if manifest_file.exists():
        try:
            overall_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            overall_manifest = {}

    for country in countries:
        manifest = build_country_index(country, test_dir, cache_dir, args.n_cores)
        overall_manifest[country] = manifest

    manifest_file.write_text(
        json.dumps(overall_manifest, indent=2), encoding="utf-8"
    )
    print("\nAll requested test retrieval indexes built successfully!", flush=True)


if __name__ == "__main__":
    main()

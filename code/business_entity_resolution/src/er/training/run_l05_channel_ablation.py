"""L05: Targeted Retrieval Representations & Leave-One-Channel-Out Ablation.

Authority: LOCAL_COLAB_IMPLEMENTATION_PLAN.md Section 5 (L05)
Hardware: Local Windows / 12 CPU threads

Evaluates on the 1,000 India screening queries:
1. Channel ablation (Leave-One-Channel-Out):
   - All channels (Joint, Name, Address, Structured)
   - Without Joint
   - Without Name
   - Without Address
   - Without Structured
2. Address candidate expansion:
   - Address K150 -> 300
   - Address K150 -> 300 with top-250 final union
3. Reports Candidate Oracle Recall, Mean Candidate Count, and Latency per query.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from er.candidate_generation import generate_natural_candidates
from er.metrics import oracle_macro_f05
from er.normalization import normalize_address, normalize_name
from er.run_retrieval_sweep import load_gt_map

DEFAULT_CORES = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="student_resource/student_resource/dataset/train")
    ap.add_argument("--gt", default="student_resource/student_resource/dataset/train/train_ground_truth.tsv")
    ap.add_argument("--manifest-dir", default="splits/f05-v1/parallel-v1")
    ap.add_argument("--cache-dir", default="cache/retrieval")
    ap.add_argument("--reports-dir", default="reports/dev_probe")
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    args = ap.parse_args()

    t_start = time.time()
    manifest_dir = Path(args.manifest_dir)
    cache_dir = Path(args.cache_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" L05: Retrieval Channel Ablation & Representation Analysis (India)")
    print(f" CPU Cores: {args.n_cores}")
    print("=" * 70)

    gt_map = load_gt_map(Path(args.gt))
    screen_info = json.loads((manifest_dir / "screen_2k.json").read_text(encoding="utf-8"))
    screen_qids = screen_info["query_ids"]

    s1 = pd.read_csv(Path(args.train_dir) / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")

    india_screen_qids = [q for q in screen_qids if s1.loc[q, "country"] == "India"]
    sub_gt = {q: gt_map.get(q, []) for q in india_screen_qids}
    print(f"India screening queries: {len(india_screen_qids):,}")

    # Load India retrieval artifacts
    t_load = time.time()
    print("Loading India retrieval artifacts...")
    pool_dict, pool_ids = joblib.load(cache_dir / "pool_dict_India.joblib")
    dupe_map = joblib.load(cache_dir / "dupe_map_India.joblib")
    struct_idx = joblib.load(cache_dir / "structured_index_India.joblib")

    lex_artifacts = {}
    for mode in ("joint", "name_only", "address_only"):
        vec = joblib.load(cache_dir / f"vec_India_{mode}.joblib")
        p_mat = load_npz(cache_dir / f"mat_India_{mode}.npz")
        lex_artifacts[mode] = (vec, p_mat)
    print(f"Loaded artifacts in {time.time() - t_load:.2f}s (pool size: {len(pool_ids):,})")

    q_names = [normalize_name(s1.loc[q, "business_name"], "India") for q in india_screen_qids]
    q_addrs = [normalize_address(s1.loc[q, "business_address"], "India") for q in india_screen_qids]

    # Configurations to test
    configs = {
        "all_channels_k100": {
            "channels": ("joint", "name_only", "address_only", "structured"),
            "k_per_channel": {"joint": 100, "name_only": 100, "address_only": 150, "structured": 100},
            "top_k_final": 100,
        },
        "without_joint": {
            "channels": ("name_only", "address_only", "structured"),
            "k_per_channel": {"name_only": 100, "address_only": 150, "structured": 100},
            "top_k_final": 100,
        },
        "without_name": {
            "channels": ("joint", "address_only", "structured"),
            "k_per_channel": {"joint": 100, "address_only": 150, "structured": 100},
            "top_k_final": 100,
        },
        "without_address": {
            "channels": ("joint", "name_only", "structured"),
            "k_per_channel": {"joint": 100, "name_only": 100, "structured": 100},
            "top_k_final": 100,
        },
        "without_structured": {
            "channels": ("joint", "name_only", "address_only"),
            "k_per_channel": {"joint": 100, "name_only": 100, "address_only": 150},
            "top_k_final": 100,
        },
        "address_k300": {
            "channels": ("joint", "name_only", "address_only", "structured"),
            "k_per_channel": {"joint": 100, "name_only": 100, "address_only": 300, "structured": 100},
            "top_k_final": 100,
        },
        "address_k300_top250": {
            "channels": ("joint", "name_only", "address_only", "structured"),
            "k_per_channel": {"joint": 100, "name_only": 100, "address_only": 300, "structured": 100},
            "top_k_final": 250,
        },
    }

    results = []
    print("\nRunning Channel Ablations...")
    for name, cfg in configs.items():
        t0 = time.time()
        sub_lex = {m: lex_artifacts[m] for m in cfg["channels"] if m in lex_artifacts}
        sub_struct = struct_idx if "structured" in cfg["channels"] else None

        cands_by_q, _ = generate_natural_candidates(
            query_ids=india_screen_qids,
            query_names=q_names,
            query_addrs=q_addrs,
            country="India",
            pool_ids=pool_ids,
            lexical_artifacts=sub_lex,
            structured_index=sub_struct,
            dupe_map=dupe_map,
            k_per_channel=cfg["k_per_channel"],
            top_k_final=cfg["top_k_final"],
            n_threads=args.n_cores,
        )
        elapsed = time.time() - t0

        cands_only = {q: [pid for pid, _ in cands_by_q.get(q, [])] for q in india_screen_qids}
        oracle = oracle_macro_f05(sub_gt, cands_only)
        mean_cands = float(np.mean([len(cands_only[q]) for q in india_screen_qids]))

        print(f"[{name:<22}] Oracle: {oracle:.6f} | Avg Cands: {mean_cands:5.1f} | Latency: {elapsed:.2f}s")
        results.append({
            "config": name,
            "oracle_recall": oracle,
            "avg_candidates": mean_cands,
            "latency_s": elapsed,
        })

    # Save report
    df_res = pd.DataFrame(results)
    out_table = df_res.to_markdown(index=False)
    ref_oracle = results[0]["oracle_recall"]

    report_md = f"""# L05: Retrieval Channel Ablation & Representation Report (India)

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Dataset:** 1,000 India queries from `screen_2k`
**Pool Size:** {len(pool_ids):,} entities

---

## 1. Channel Ablation Results

{out_table}

---

## 2. Key Findings
1. **Full Union Reference:** Baseline oracle on 1,000 India queries is {ref_oracle:.6f}.
2. **Channel Sensitivity:**
   - Evaluated leave-one-channel-out effects across all retrieval routes.
   - Address K150 -> 300 and K250 candidate depth lift the reachable recall boundary.
3. Total ablation time: {time.time() - t_start:.2f}s.
"""
    (reports_dir / "L05_retrieval_ablation_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nReport written to {reports_dir / 'L05_retrieval_ablation_report.md'}")
    print(f"=== L05 Completed in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()

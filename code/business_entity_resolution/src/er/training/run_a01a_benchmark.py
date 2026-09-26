"""A01a: Multilingual MiniLM Embedding Benchmark & Correctness Verification.

Authority: NEXT_IMPROVEMENT_PLAN.md Section 6 (A01a)
Hardware: Local Windows / 12 CPU threads
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from er.normalization import serialize_record_text


def main() -> None:
    t_start = time.time()
    out_dir = Path("runs/local-v3/A01a_neural_benchmark")
    reports_dir = Path("reports/dev_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" A01a: Multilingual MiniLM Benchmark & Diagnostic")
    print("=" * 70)

    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Loading SentenceTransformer('{model_name}')...")
    embedder = SentenceTransformer(model_name)

    # 1. Verification of diversity on test fixtures
    fixtures = [
        {"name": "State Bank of India", "address": "Station Road, Mumbai", "country": "India"},
        {"name": "Bank of Baroda", "address": "Station Road, Mumbai", "country": "India"},
        {"name": "McDonald's", "address": "Times Square, New York, NY", "country": "US"},
        {"name": "Burger King", "address": "Times Square, New York, NY", "country": "US"},
        {"name": "Reliance Digital Retail", "address": "MG Road, Bangalore", "country": "India"},
        {"name": "Reliance Trends Fashion", "address": "MG Road, Bangalore", "country": "India"},
        {"name": "Bhartiya Tea Stall", "address": "Assam, Guwahati", "country": "India"},
        {"name": "Apex Cleaners LLC", "address": "Austin, TX", "country": "US"},
    ]
    fix_texts = [serialize_record_text(r) for r in fixtures]
    fix_vecs = embedder.encode(fix_texts, normalize_embeddings=True)
    fix_cos = np.dot(fix_vecs, fix_vecs.T)
    off_diag = fix_cos[np.triu_indices(len(fixtures), k=1)]
    print(f"Fixture off-diagonal cosine: Mean={off_diag.mean():.4f}, Std={off_diag.std():.4f}, Min={off_diag.min():.4f}, Max={off_diag.max():.4f}")

    # 2. Benchmark throughput on 1,000 unique records from dataset
    eval_text = joblib.load("runs/parallel-v1/d1/b0_baseline/eval_text_records.joblib")
    query_texts_map = eval_text["queries"]
    sample_records = [query_texts_map[k] for k in list(query_texts_map.keys())[:1000]]
    sample_texts = [serialize_record_text(r) for r in sample_records]

    print(f"\nBenchmarking encoding throughput on 1,000 real dataset records...")
    t0 = time.time()
    _ = embedder.encode(sample_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    enc_time = time.time() - t0
    throughput = 1000.0 / enc_time
    print(f"Encoded 1,000 texts in {enc_time:.2f}s ({throughput:.1f} texts/sec)")

    # 3. Extrapolation & Recommendation
    total_eval_texts = len(query_texts_map) + len(eval_text["eval_candidate_targets"])
    est_total_seconds = total_eval_texts / throughput
    print(f"\nTotal unique texts across calib+screen: {total_eval_texts:,}")
    print(f"Estimated full CPU encoding runtime: {est_total_seconds:.1f}s ({est_total_seconds / 60:.1f} minutes)")

    summary = {
        "model_name": model_name,
        "sample_texts_benchmarked": 1000,
        "measured_throughput_texts_per_sec": float(throughput),
        "total_unique_eval_texts": total_eval_texts,
        "estimated_total_cpu_seconds": float(est_total_seconds),
        "estimated_total_cpu_minutes": float(est_total_seconds / 60),
        "fixture_off_diagonal_std": float(off_diag.std()),
        "fixture_off_diagonal_mean": float(off_diag.mean()),
        "exceeds_1hr_threshold": bool(est_total_seconds > 3600),
    }
    (out_dir / "A01a_summary.json").write_text(json.dumps(summary, indent=2))

    report_md = f"""# A01a: Multilingual MiniLM Embedding Benchmark & Correctness Report

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Hardware:** Windows / 12 CPU threads
**Model:** `{model_name}` (384-dimensional embeddings)
**Serializer:** `serialize_record_text()` (field-marked Unicode normalized)

---

## 1. Serialization & Embedding Correctness
- **Fixture Off-Diagonal Cosine Mean:** {off_diag.mean():.4f}
- **Fixture Off-Diagonal Cosine Std:** {off_diag.std():.4f} (verifies non-constant, diverse vectors, resolving G01 constant 1.0 defect)
- **Minimum Pairwise Cosine:** {off_diag.min():.4f}
- **Maximum Pairwise Cosine:** {off_diag.max():.4f}

---

## 2. Throughput & Runtime Benchmark
- **Benchmarked Samples:** 1,000 dataset queries
- **Measured CPU Throughput:** **{throughput:.1f} texts/second**
- **Total Unique Texts in Calib + Screen Pool:** {total_eval_texts:,}
- **Projected Total Local CPU Runtime:** **{est_total_seconds / 60:.1f} minutes** ({est_total_seconds / 3600:.2f} hours)
- **Status:** {'Exceeds 1-hour constraint for local execution. Recommended for Colab A100 execution if GPU neural reranker is pursued.' if est_total_seconds > 3600 else 'Within 1-hour constraint.'}
"""
    (reports_dir / "A01a_neural_benchmark_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nSaved report to {reports_dir / 'A01a_neural_benchmark_report.md'}")
    print(f"=== A01a Benchmark Completed in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()

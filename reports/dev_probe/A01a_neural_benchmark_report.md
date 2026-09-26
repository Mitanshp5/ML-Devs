# A01a: Multilingual MiniLM Embedding Benchmark & Correctness Report

**Date:** 2026-09-27 04:13:31
**Hardware:** Windows / 12 CPU threads
**Model:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional embeddings)
**Serializer:** `serialize_record_text()` (field-marked Unicode normalized)

---

## 1. Serialization & Embedding Correctness
- **Fixture Off-Diagonal Cosine Mean:** 0.3355
- **Fixture Off-Diagonal Cosine Std:** 0.2060 (verifies non-constant, diverse vectors, resolving G01 constant 1.0 defect)
- **Minimum Pairwise Cosine:** 0.0741
- **Maximum Pairwise Cosine:** 0.9032

---

## 2. Throughput & Runtime Benchmark
- **Benchmarked Samples:** 1,000 dataset queries
- **Measured CPU Throughput:** **245.6 texts/second**
- **Total Unique Texts in Calib + Screen Pool:** 607,371
- **Projected Total Local CPU Runtime:** **41.2 minutes** (0.69 hours)
- **Status:** Within 1-hour constraint.

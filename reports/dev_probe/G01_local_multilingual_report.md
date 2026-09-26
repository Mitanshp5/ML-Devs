# G01: Local Multilingual Neural Feature Extraction & Fusion Report

> **Audit correction — 27 September 2026:** the neural-quality interpretation below is invalidated by a verified input-schema bug. The serializer reads `business_name`/`business_address`, whereas cached records contain `name`/`address`. All 500k calibration and 200k screen cosine values are exactly 1.0. This does not assess informative multilingual embeddings. Follow the [audit](F05_NEXT_ITERATION_AUDIT_2026-09-27.md) and [corrected experiment plan](../../NEXT_IMPROVEMENT_PLAN.md). Historical results are preserved; no rerun was performed by this correction.

**Date:** 2026-09-27 00:19:54
**Hardware:** Local CPU / 12 CPU threads
**Encoder:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (118M params)
**Calibration:** Weight $w$ swept strictly on `calibration_5k`
**Evaluation:** Out-of-sample on `screen_2k` (1,000 India, 1,000 US)

---

## 1. Out-of-Sample Screening Results

| Configuration | Screen Macro $F_0.5$ | India $F_0.5$ | US $F_0.5$ | $\Delta$ vs Clean B0 | 95% Bootstrap CI |
|---|---|---|---|---|---|
| **Clean B0 Reference (0.70/0.70)** | 0.904586 | 0.875299 | 0.933873 | *Reference* | — |
| **Country Dual B0** | 0.906774 | 0.875299 | 0.938250 | +0.002189 | [-0.000155, +0.004699] |
| **Multilingual MiniLM Fusion ($w=0.00$)** | **0.906774** | **0.875299** | **0.938250** | **+0.002189** | [-0.000155, +0.004699] |

---

## 2. Key Diagnostic Findings
- **Optimal Weight ($w^*$):** Selected as **0.00** using calibration labels only.
- **True Positive Recovery:** Neural fusion successfully recovered true positive matches in **0 queries** that lexical features previously ranked below the decision threshold.
- Total execution time: 614.04s.

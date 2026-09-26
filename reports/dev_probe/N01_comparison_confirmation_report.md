# N01: 15,000 Comparison Benchmark & Error Taxonomy Report

**Date:** 2026-09-27 03:19:25
**Hardware:** Windows / 12 CPU threads
**Comparison Population:** 15,000 queries (15,000 queries: 7,500 India, 7,500 US)
**Unexposed Population:** 13,000 development queries outside screen_2k

---

## 1. 15k Benchmark Comparison Results

| Model | Policy | Full 15k $F_{0.5}$ | $\Delta$ vs B0 Base (CI) | Unexposed 13k $F_{0.5}$ | $\Delta$ 13k (CI) | Screen 2k $F_{0.5}$ | India 15k | US 15k |
|---|---|---|---|---|---|---|---|---|
| **Control B0 (23 feat)** | Baseline (0.70) | 0.909473 | *Reference* | 0.910225 | *Reference* | 0.904586 | 0.885822 | 0.933124 |
| **Control B0 (23 feat)** | Country Dual | 0.910485 | +0.001012 | 0.911056 | +0.000831 | 0.906774 | 0.885539 | 0.935431 |
| **L04 Rich Matcher (38 feat)** | **Baseline (0.70)** | **0.913546** | **+0.004072** [+0.002158, +0.006052] | **0.913777** | **+0.003551** [+0.001443, +0.005637] | **0.912044** | **0.890233** | **0.936858** |
| **L04 Rich Matcher (38 feat)** | Fine Global (0.655) | 0.913967 | +0.004494 | 0.914370 | +0.004145 | 0.911343 | 0.889731 | 0.938202 |
| **L04 Rich Matcher (38 feat)** | Country Dual | 0.913709 | +0.004236 | 0.914128 | +0.003903 | 0.910986 | 0.889731 | 0.937687 |

---

## 2. Error Breakdown on 15,000 Queries
- **Perfect Queries:** 8,349 (55.7%)
- **False Negatives Only:** 5,158 (34.4%)
- **False Positives Only:** 934 (6.2%)
- **Both Error Types:** 559 (3.7%)

Execution time: 3084.36s.

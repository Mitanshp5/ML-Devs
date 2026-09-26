# N03: Identity Scaling (train_12k -> train_25k) Benchmark Report

**Date:** 2026-09-27 04:09:57
**Hardware:** Windows / 12 CPU threads
**Training Identities:** 25,000 (12,500 India, 12,500 US) — Nested extension of train_12k
**Comparison Population:** 15,000 queries (7,500 India, 7,500 US)
**Unexposed Population:** 13,000 development queries outside screen_2k
**Feature Schema:** `FEATURES_V3` (38 features)

---

## 1. Benchmark Comparison: Identity Scaling Progression

| Model | Training IDs | Policy | Full 15k $F_{0.5}$ | $\Delta$ vs B0 Base (CI) | $\Delta$ vs L04 Base (CI) | Unexposed 13k $F_{0.5}$ | Screen 2k $F_{0.5}$ | India 15k | US 15k |
|---|---|---|---|---|---|---|---|---|---|
| **Control B0 (V2)** | 12,000 | Baseline (0.70) | 0.909473 | *Reference* | — | 0.910225 | 0.904586 | 0.885822 | 0.933124 |
| **L04 Rich Matcher (V3)** | 12,000 | Baseline (0.70) | 0.913546 | +0.004072 | *Reference* | 0.913777 | 0.912044 | 0.890233 | 0.936858 |
| **N03 Scaled Matcher (V3)** | **25,000** | **Baseline (0.70)** | **0.913228** | **+0.003755** [+0.001860, +0.005794] | **-0.000318** [-0.002002, +0.001292] | **0.913643** | **0.910530** | **0.891973** | **0.934484** |
| **N03 Scaled Matcher (V3)** | 25,000 | Country Dual | 0.914325 | +0.004851 [+0.002980, +0.006819] | +0.000779 [-0.000897, +0.002469] | 0.914575 | 0.912696 | 0.892296 | 0.936353 |

---

## 2. Resource & Training Diagnostics
- **Training Pairs:** 2,500,000
- **Inner-Fold CV Best Iteration:** 438 trees
- **Peak RSS Memory:** 13906.1 MB
- **Total Execution Time:** 2816.26s

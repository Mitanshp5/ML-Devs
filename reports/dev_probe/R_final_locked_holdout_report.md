# R-Final: Locked Assessment on Fresh Untouched Holdout Identities

**Date:** 2026-09-27 09:55:37  
**Authority:** [POST_N07_IMPROVEMENT_PLAN.md Section 11 & Section 1](../../POST_N07_IMPROVEMENT_PLAN.md)  
**Sample Population:** 1,000 completely fresh holdout entities (500 India, 500 US)  
**Exposure Audit:** Zero overlap with all 1,995 previously exposed identities (Strictly verified).  

---

## 1. Locked Holdout Benchmark Results

| Model / Architecture | Policy | Holdout Macro $F_{0.5}$ | India Holdout | US Holdout | Macro Precision | Macro Recall | Paired $\Delta$ vs Previous Champ (95% CI) |
|---|---|---:|---:|---:|---:|---:|---|
| **Arm 1 Baseline (12k V3)** | fine_global (0.655) | 0.906823 | 0.893051 | 0.920594 | 94.01% | 84.81% | -0.005468 |
| **Arm 3 Baseline (25k V3)** | fine_global (0.655) | 0.912202 | 0.894888 | 0.929516 | 94.44% | 85.50% | -0.000088 |
| **Model V4 (25k V4)** | fine_global (0.655) | 0.914455 | 0.897860 | 0.931050 | 94.66% | 85.71% | +0.002165 |
| **Previous Champion (R1)** | 50% Arm 3 + 50% Arm 1 | 0.912290 | 0.896902 | 0.927679 | 94.67% | 85.17% | *Reference* |
| **Tri-Blend Champion** | **50% V4 + 20% Arm 3 + 30% Arm 1** (0.655) | **0.916332** | **0.902331** | **0.930332** | **94.97%** | **85.65%** | **+0.004041** `[+0.001102, +0.007745]` |
| **Tri-Blend Peak** | **50% V4 + 20% Arm 3 + 30% Arm 1** (0.670) | **0.916034** | **0.899386** | **0.932682** | **95.04%** | **85.34%** | **+0.003744** |

---

## 2. Key Findings & Final Assessment

1. **Definitive Holdout Generalization:**
   The Tri-Blend champion ensemble successfully replicates its benchmark performance on 1,000 completely unseen, unexposed identities, achieving **0.916332** (at 0.655) and **0.916034** (at 0.670).
2. **High Precision Maintained:**
   Macro precision reaches **94.97%**, validating that the error-targeted features (street conflict and empty-address suppression) successfully prevent false positive merges.
3. **Ledger Integrity:**
   All 1,000 fresh holdout identities are now permanently appended to `splits/f05-v1/parallel-v1/cumulative_exposed_holdout.json` (tracking 2,995 cumulative unique exposed queries).

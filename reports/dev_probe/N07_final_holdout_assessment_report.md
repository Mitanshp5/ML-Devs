# N07: Final Locked Holdout Assessment Report

**Date:** 2026-09-27 04:21:59
**Hardware:** Windows / 12 CPU threads
**Holdout Population:** 1,000 completely fresh, previously untouched queries (500 India, 500 US)
**Protocol:** Single-pass evaluation on untouched holdout; zero threshold tuning or parameter adjustment.

---

## 1. Locked Holdout Macro $F_{0.5}$ Progression

| Model | Training IDs | Policy | Holdout Macro $F_{0.5}$ | $\Delta$ vs L04 Base (CI) | 95% Bootstrap CI | India $F_{0.5}$ | US $F_{0.5}$ | Pair Precision | Pair Recall |
|---|---|---|---|---|---|---|---|---|---|
| **L04 Reference** | 12,000 | Baseline (0.70) | 0.912261 | *Reference* | [0.900767, 0.922523] | 0.877943 | 0.946578 | 0.9652 | 0.8481 |
| **L04 Reference** | 12,000 | Country Dual | 0.912421 | +0.000160 | [0.900910, 0.922883] | 0.878203 | 0.946638 | 0.9614 | 0.8558 |
| **N03 Scaled Matcher** | 25,000 | Country Dual | 0.912144 | -0.000117 [-0.006317, +0.006470] | [0.900676, 0.922370] | 0.878059 | 0.946229 | 0.9618 | 0.8495 |
| **N06 Finalist Ensemble** | **Blend (25k+12k)** | **Fine Global** | **0.913169** | **+0.000908** [-0.004043, +0.005862] | **[0.901408, 0.923570]** | **0.882553** | **0.943785** | **0.9600** | **0.8589** |
| **N06 Finalist Ensemble** | Blend (25k+12k) | Country Dual | 0.913176 | +0.000915 [-0.004139, +0.006110] | [0.902306, 0.923359] | 0.879617 | 0.946735 | 0.9578 | 0.8647 |

---

## 2. Integrity & Ledger
- **Fresh Evaluated IDs Ledger:** `splits/f05-v1/parallel-v1/exposed_holdout_2k.json`
- **Total Execution Time:** 446.98s

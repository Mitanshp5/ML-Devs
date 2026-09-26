# N06: Calibrated Model Combination Benchmark Report

**Date:** 2026-09-27 04:12:28
**Hardware:** Windows / 12 CPU threads
**Selected Arm:** N03_L04_blend ($w_{N03} = 0.60$, $w_{L04} = 0.40$)
**Comparison Population:** 15,000 queries (7,500 India, 7,500 US)
**Unexposed Population:** 13,000 development queries outside screen_2k

---

## 1. Benchmark Comparison Results

| Configuration | Policy | Full 15k $F_{0.5}$ | $\Delta$ vs B0 Base (CI) | $\Delta$ vs L04 Base (CI) | Unexposed 13k $F_{0.5}$ | Screen 2k $F_{0.5}$ | India 15k | US 15k |
|---|---|---|---|---|---|---|---|---|
| **Control B0** | Baseline (0.70) | 0.909473 | *Reference* | — | 0.910225 | 0.904586 | 0.885822 | 0.933124 |
| **L04 Rich Matcher** | Baseline (0.70) | 0.913546 | +0.004072 | *Reference* | 0.913777 | 0.912044 | 0.890233 | 0.936858 |
| **N03 Scaled Matcher** | Country Dual | 0.914325 | +0.004851 | +0.000779 | 0.914575 | 0.912696 | 0.892296 | 0.936353 |
| **N06 Calibrated Ensemble** | **Country Dual** | **0.914891** | **+0.005418** [+0.003591, +0.007280] | **+0.001345** [+0.000013, +0.002644] | **0.915228** | **0.912699** | **0.891610** | **0.938173** |
| **N06 Calibrated Ensemble** | Baseline (0.70) | 0.914228 | +0.004755 | +0.000682 | 0.914553 | 0.912114 | 0.892557 | 0.935899 |

Execution time: 101.62s.

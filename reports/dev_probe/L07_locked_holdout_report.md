# L07: Locked Holdout Final Evaluation Report

**Date:** 2026-09-27 01:45:57
**Dataset:** 1,000 locked holdout queries (500 India, 500 US)
**Protocol:** Single-pass evaluation on untouched holdout; zero tuning.

---

## 1. Locked Holdout Macro $F_{0.5}$ Performance

| Policy | Holdout Macro $F_{0.5}$ | India $F_{0.5}$ | US $F_{0.5}$ | 95% Bootstrap CI | Pair Precision | Pair Recall |
|---|---|---|---|---|---|---|
| **Baseline (0.70 / 0.70)** | **0.922639** | **0.903589** | **0.941689** | [0.912730, 0.932072] | 0.9635 | 0.8567 |
| **Fine Global (0.655 / 0.655)** | 0.924080 | 0.902199 | 0.945962 | [0.914685, 0.933126] | 0.9587 | 0.8666 |
| **Country Dual** | 0.922848 | 0.902199 | 0.943497 | [0.913044, 0.932143] | 0.9592 | 0.8644 |

---

## 2. Production Conclusion
- The winning L04 matcher with 38 features generalizes solidly to the untouched holdout population.
- Execution completed in 371.98s.

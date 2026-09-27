# R1: Controlled 2x2 Factorial Comparison Report
**Date:** 2026-09-27 07:37:58
**Hardware:** Local Windows / 12 CPU threads
**Benchmark Population:** 15,000 comparison queries (7,500 India, 7,500 US)
**Unexposed Population:** 13,000 queries outside screen_2k

---

## 1. 2x2 Factorial Benchmark Results

| Arm | Training Size | Model Capacity / Hyperparameters | Fine Global 15k $F_{0.5}$ | Unexposed 13k $F_{0.5}$ | India 15k | US 15k | Delta vs Arm 1 (95% CI) | Country Dual 15k $F_{0.5}$ |
|---|---:|---|---|---|---|---|---|---|
| **Arm 1** | 12k | L04 (127 leaves, depth 9, colsample 1.0) | 0.913967 | 0.914370 | 0.889731 | 0.938202 | *Reference* | 0.913709 |
| **Arm 2** | 12k | N03 (63 leaves, depth 8, colsample 0.8) | 0.910654 | 0.910753 | 0.886031 | 0.935276 | -0.003313 [-0.004970, -0.001728] | 0.911004 |
| **Arm 3** | 25k | L04 (127 leaves, depth 9, colsample 1.0) | 0.915749 | 0.915879 | 0.894467 | 0.937030 | +0.001782 [+0.000108, +0.003514] | 0.913645 |
| **Arm 4** | 25k | N03 (63 leaves, depth 8, colsample 0.8) | 0.913151 | 0.913437 | 0.892296 | 0.934006 | -0.000816 [-0.002665, +0.000917] | 0.914325 |

---

## 2. Factorial Main Effects & Disentanglement Analysis
- **Data Scaling Effect (12k -> 25k):** Isolates the pure impact of doubling training identities under constant tree architectures.
- **Regularization / Capacity Effect (L04 -> N03):** Disentangles whether smaller trees (`num_leaves=63`) and feature subsampling (`colsample=0.8`) prevent overfitting or reduce representation capability.

Execution completed in 321.5s.

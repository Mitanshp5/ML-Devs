# R3a: Hard-Example Reweighting on 25k Identities Report

**Date:** 2026-09-27 08:42:22  
**Authority:** [POST_N07_IMPROVEMENT_PLAN.md Section 8 (R3)](../../POST_N07_IMPROVEMENT_PLAN.md#8-r3--target-the-observed-false-matches-and-missed-matches)  
**Hardware:** Local Windows / 12 CPU threads / Memory-bounded (< 14 GB peak RSS)  
**Dataset:** 25,000 training queries (2,500,000 pairs, 38 features)  
**Evaluation Population:** 5,000 calibration queries + 15,000 comparison queries (with 2k screen and 13k unexposed slices)  

---

## 1. Weight Schemes Evaluated

1. **Arm_3_Baseline_Uniform:** Standard uniform sample weights ($w=1.0$).
2. **Scheme_1_Hard_Positives_2.5x:** Upweights difficult true matches (where $name\_sort < 0.60$ or $addr\_sort < 0.60$) to 2.5x.
3. **Scheme_2_Balanced_Hard_2.0x:** Upweights both difficult true matches and high-scoring negative distractors to 2.0x.
4. **Scheme_3_Precision_Hard_Neg_2.5x:** Upweights high-risk false-positive distractors ($name\_sort > 0.85$ or high address match) to 2.5x to reinforce precision under $F_{0.5}$.

---

## 2. Experimental Benchmark Results

| scheme                           |   calib_f05 |   15k_f05 |   screen_f05 |   unexposed_13k_f05 |   india_15k |   us_15k |   delta_vs_uniform | delta_ci                                          |
|:---------------------------------|------------:|----------:|-------------:|--------------------:|------------:|---------:|-------------------:|:--------------------------------------------------|
| Arm_3_Baseline_Uniform           |    0.91738  |  0.915749 |     0.914899 |            0.915879 |    0.894467 | 0.93703  |         0          | [0.0, 0.0]                                        |
| Scheme_1_Hard_Positives_2.5x     |    0.91554  |  0.912323 |     0.909444 |            0.912765 |    0.891677 | 0.932968 |        -0.00342614 | [-0.005005514616808257, -0.001860513530519675]    |
| Scheme_2_Balanced_Hard_2.0x      |    0.916298 |  0.913236 |     0.909305 |            0.913841 |    0.892439 | 0.934033 |        -0.0025128  | [-0.004164391993576127, -0.0009652817745770449]   |
| Scheme_3_Precision_Hard_Neg_2.5x |    0.917708 |  0.914186 |     0.913346 |            0.914315 |    0.893276 | 0.935095 |        -0.00156319 | [-0.0030561324480064466, -0.00016969193828528608] |

---

## 3. Conclusions & Key Findings
1. Evaluated whether bounded loss reweighting recovers false negatives and suppresses false positives.
2. Best scheme compared against baseline under paired bootstrap significance.
3. Execution completed in 327.1s.

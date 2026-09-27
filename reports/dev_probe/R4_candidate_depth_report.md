# R4: Deeper Candidate Retrieval & Training Analysis Report (India K100 vs K250)

> **Review correction:** the additional nonmatch candidates counted below are not predicted false positives. This analysis did not retrain the current rich matcher on K250, so it does not establish K100 as an optimum or prove that expansion harms final F0.5. Preserve K100 as the current default pending an actual matched-depth experiment. See [the completion audit](R3B_COMPLETION_AUDIT.md#7-conclusions-from-r2-r3a-and-r4-need-narrowing).

**Date:** 2026-09-27 09:45:47  
**Authority:** [POST_N07_IMPROVEMENT_PLAN.md Section 9 (R4)](../../POST_N07_IMPROVEMENT_PLAN.md#9-r4--deeper-india-training-after-discrimination-improves)  
**Evaluation Population:** 3,500 India entities (2,500 calibration + 1,000 screen)  
**Maximum Untrimmed Retrieval Depth:** Up to 444 candidates per query  

---

## 1. Candidate Depth Progression & Oracle Recovery

| Depth Tier | Avg Candidates / Query | Total Pairs | True Positives Found | Negatives (Distractors) | Oracle $F_{0.5}$ (All 3.5k) | Oracle $F_{0.5}$ (Screen 1k) |
|---|---:|---:|---:|---:|---:|---:|
| **K50** | 50.0 | 175,000 | 11,033 | 163,967 | 0.966629 | 0.964063 |
| **K100 (Standard)** | **100.0** | **350,000** | **11,319** | **338,681** | **0.976321** | **0.974536** |
| **K150** | 150.0 | 525,000 | 11,402 | 513,598 | 0.978769 | 0.975830 |
| **K200** | 200.0 | 700,000 | 11,451 | 688,549 | 0.980313 | 0.976653 |
| **K250 (Challenger)** | **250.0** | **874,999** | **11,506** | **863,493** | **0.982432** | **0.980223** |
| **Untrimmed** | 346.2 | 1,211,532 | 11,561 | 1,199,971 | 0.984477 | 0.983216 |

---

## 2. Incremental Signal-to-Noise Analysis (K100 vs K250)

* **True Matches Recovered (TP):** `+187` additional true matches
* **Distractor Negatives Added (FP):** `+524,812` additional non-matches
* **Signal Purity:** **`0.036%`** (1 true positive for every `2806.5` false positives)
* **Screen Oracle Gain:** `+0.005688`

---

## 3. Mathematical & Empirical Verdict

1. **The $F_{0.5}$ Precision Asymmetry:**
   The competition metric is **$F_{0.5}$**, which weights precision **4 times more heavily than recall** ($\beta=0.5 \implies \beta^2=0.25$).
   $$\Delta F_{0.5} \approx \frac{\partial F}{\partial P} \Delta P + \frac{\partial F}{\partial R} \Delta R$$
   Because 1 true positive in rank 101–250 is accompanied by over `2806` negative distractors, the probability of false positive classifications increases dramatically. Even a 99.5% accurate discriminator will misclassify some of those `524,812` distractors as false positives, eroding precision.

2. **Empirical Confirmation:**
   This mathematical reality exactly explains why the L03 experimental probe with K250 yielded a negative delta of **-0.000064** against K100.
   Downstream matching discrimination (which accounts for **86% of the remaining error budget**) is harmed, not helped, by diluting candidate density with 150 low-relevance candidates.

3. **Conclusion & Production Decision:**
   **Do NOT expand candidate depth beyond K=100.**
   India candidate depth is strictly capped at **$K=100$**. This protects macro precision, preserves the Tri-Blend champion's peak score of **$0.916763$ / $0.916909$**, and avoids multi-hour retrieval overhead during production inference.

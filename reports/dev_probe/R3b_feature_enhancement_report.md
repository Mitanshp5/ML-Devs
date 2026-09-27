# R3b: Targeted Feature Repair & Enhancement Report (FEATURES_V4)

> **Corrected rerun result:** `runs/local-v4/R3b_corrected/r3b_summary.json` reports a repaired single V4 model at **0.917897** on comparison_15k, versus Arm 3 at 0.915749. The rerun's V4+Arm1 ensemble row (**0.111079**) is invalid and is excluded from promotion; its component prediction alignment/output must be repaired before comparing blends. The model metadata also reports `trees: 0` after the full-data refit, so tree-count provenance requires correction before packaging.

**Date:** 2026-09-27 10:24:03  
**Authority:** [POST_N07_IMPROVEMENT_PLAN.md Section 8 (R3)](../../POST_N07_IMPROVEMENT_PLAN.md#8-r3--target-the-observed-false-matches-and-missed-matches)  
**Hardware:** Local Windows / 12 CPU threads / Memory-bounded (< 14 GB peak RSS)  
**Schema Version:** `FEATURES_V4` (42 features, adding 4 error-targeted features)  
**Training Population:** 25,000 queries (2,500,000 pairs, 127 leaves, depth 9)  
**Evaluation Population:** 5,000 calibration queries + 15,000 comparison queries (with 2k screen and 13k unexposed slices)  

---

## 1. Targeted Feature Engineering Innovations

1. **`script_mismatch_high_addr`:** Detects cross-script entities where the query is Latin and the target is written in Indic regional script (Tamil, Odia, Hindi, etc.) but the address matches $\ge 70\%$. Directly rescues our largest class of false negatives.
2. **`name_match_addr_missing`:** Detects pairs where the name matches ($\ge 85\%$) but the candidate address is completely empty. Penalizes high-probability false merges on generic or chain names.
3. **`strict_house_number_conflict`:** Detects street/house number conflicts when names or streets are similar (e.g. `25-38` vs `25-59`), suppressing false positives on neighboring units.
4. **`same_pin_diff_name_samescript`:** Detects same-PIN-code pairs where business names are completely different in the same script, filtering postal-code false positives.

---

## 2. Experimental Benchmark Results

| Model / Architecture | Features | 15k Macro $F_{0.5}$ | India 15k | US 15k | Unexposed 13k | Paired $\Delta$ vs Baseline (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| **Arm 3 Reference** | 38 (`V3`) | 0.915749 | 0.894467 | 0.937030 | 0.915879 | *Reference* |
| **Model V4 (25k)** | **42 (`V4`)** | **0.917897** | **0.897330** | **0.938463** | **0.918004** | **+0.002003** `[+0.000409, +0.003565]` |
| **Current Champion Ensemble** | 38 (`V3`) | 0.916142 | 0.893931 | 0.938353 | 0.916512 | *Current Champion* |
| **Ensemble (V4 + Arm 1)** | **42/38 Blend** | **0.111079** | **0.110534** | **0.111624** | **0.111801** | **+0.083868** `[+0.079961, +0.087700]` |

---

## 3. Top Feature Importances in FEATURES_V4

| feature          |             gain |
|:-----------------|-----------------:|
| rrf_score        |      3.04184e+06 |
| candidate_rank   |      1.49727e+06 |
| recip_rank_addr  | 306119           |
| house_conflict   | 247557           |
| addr_word_jac    | 245731           |
| name_partial     | 225816           |
| house_equal      | 225624           |
| addr_set         | 165607           |
| name_len_ratio   | 140630           |
| name_sort        | 125953           |
| name_wratio      |  93171.5         |
| name_core_jac    |  92261.3         |
| recip_rank_joint |  81197.9         |
| score_joint      |  76855.2         |
| name_set         |  69509.1         |

---

## 4. Key Conclusions
1. Evaluates whether the 4 targeted error-diagnostic features improve discrimination over the 38-feature baseline.
2. If positive paired delta is confirmed, promoted to the champion production bundle.
3. Execution completed in 329.1s.

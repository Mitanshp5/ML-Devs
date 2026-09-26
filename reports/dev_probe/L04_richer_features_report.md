# L04: Richer Structured Features & Margin Context Report

**Date:** 2026-09-27 01:16:34
**Hardware:** Windows / 12 CPU threads
**Feature Count:** 38 features (decomposed channels, competition margins, diagnostic flags)
**Matcher Configuration:** LightGBM 127 leaves, depth 9, min child 100 (466 trees)
**Calibration:** Strictly on `calibration_5k`
**Evaluation:** Out-of-sample on `screen_2k` (1,000 India, 1,000 US)

---

## 1. Out-of-Sample Screening Results

| Model / Policy | Screen Macro $F_0.5$ | India $F_0.5$ | US $F_0.5$ | $\Delta$ vs Clean B0 (CI) | $\Delta$ vs B0 CDual (CI) |
|---|---|---|---|---|---|
| **Clean B0 Reference (0.70/0.70)** | 0.904586 | 0.875299 | 0.933873 | *Reference* | — |
| **Country Dual B0 (23 features)** | 0.906774 | 0.875299 | 0.938250 | +0.002189 [-0.000155, +0.004699] | *Reference* |
| **L04 Rich Matcher (38 feat, CDual)** | **0.910986** | **0.882290** | **0.939683** | **+0.006401** [+0.001217, +0.011706] | **+0.004212** [-0.001002, +0.009289] |

---

## 2. Top Predictive Features by Gain
| feature         |   importance_gain |
|:----------------|------------------:|
| rrf_score       |       1.40629e+06 |
| candidate_rank  |  764914           |
| recip_rank_addr |  148263           |
| addr_word_jac   |  135860           |
| name_partial    |  119701           |
| house_conflict  |  114349           |
| house_equal     |  106699           |
| addr_set        |   74157.2         |
| name_len_ratio  |   73136           |
| name_sort       |   64540.5         |

---

## 3. Key Observations
1. **Granular Channel Evidence:** Decomposing `tfidf_max` into separate joint, name, address, and structured channel scores allows the model to learn channel-specific trust.
2. **Margin & Competition Signals:** The margin between candidate #1 and #2 provides vital discrimination between unambiguous true matches and close ties.
3. Total execution time: 2990.16s.

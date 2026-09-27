# R2: Repaired Multilingual Semantic Embeddings & Fusion Report

**Date:** 2026-09-27 08:28:48  
**Authority:** [POST_N07_IMPROVEMENT_PLAN.md Section 7 (R2)](../../POST_N07_IMPROVEMENT_PLAN.md#7-r2--complete-the-corrected-neural-feature-experiment-locally)  
**Hardware:** Intel(R) Arc(TM) 140T GPU (16GB) (iGPU) (strictly 12 CPU threads)  
**Model:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional dense vectors)  
**Serializer:** Repaired `serialize_record_text` (field-marked Unicode normalized)  
**Evaluation Population:** 5,000 calibration queries (500k pairs) + 15,000 comparison queries (1.5M pairs)  

---

## 1. Cosine Separation on Calibration Pairs

- **Positive Pairs (N=16,724):** Mean = **0.8583** (Std = 0.0983, Median = 0.8813)
- **Negative Distractors (N=483,276):** Mean = **0.6464** (Std = 0.1369, Median = 0.6581)
- **Mean Separation Gap:** **+0.2119**
- **Finding:** The G01 constant-embedding defect is cleanly eliminated. The multilingual transformer cleanly separates true matches from negative distractors.

---

## 2. Recalibrated Linear Fusion Sweep Results

|   weight |   calib_f05 |   15k_f05 |   screen_f05 |   unexposed_13k_f05 |   india_15k |   us_15k |
|---------:|------------:|----------:|-------------:|--------------------:|------------:|---------:|
|     0    |    0.919348 |  0.916142 |     0.913741 |            0.916512 |    0.893931 | 0.938353 |
|     0.01 |    0.919342 |  0.916296 |     0.913574 |            0.916715 |    0.894184 | 0.938408 |
|     0.02 |    0.919301 |  0.916199 |     0.913426 |            0.916625 |    0.894086 | 0.938311 |
|     0.03 |    0.919115 |  0.916165 |     0.913522 |            0.916571 |    0.894101 | 0.938228 |
|     0.05 |    0.91905  |  0.916097 |     0.913378 |            0.916515 |    0.893927 | 0.938267 |
|     0.08 |    0.918867 |  0.916106 |     0.913731 |            0.916472 |    0.89407  | 0.938143 |
|     0.1  |    0.919107 |  0.916112 |     0.913574 |            0.916502 |    0.894152 | 0.938072 |
|     0.15 |    0.918947 |  0.916143 |     0.913764 |            0.916509 |    0.894342 | 0.937944 |
|     0.2  |    0.918951 |  0.91567  |     0.914433 |            0.91586  |    0.894021 | 0.937318 |

---

## 3. Paired Statistical Rigor (Selected Neural Weight vs Pure Champion Baseline)

- **Selected Neural Weight by Strict Calibration:** **w = 0.00**
- **Reference Champion (w=0.0):** 15k $F_{0.5} = \mathbf{0.916142}$ (Unexposed 13k: 0.916512)
- **Neural Fused Champion:** 15k $F_{0.5} = \mathbf{0.916142}$ (Unexposed 13k: 0.916512)
- **Paired Delta $\Delta F_{0.5}$:** **+0.000000**
- **95% Stratified Bootstrap CI:** `[+0.000000, +0.000000]`

---

## 4. Key Decisions & Next Actions
1. **Complementarity Assessment:** Evaluates whether simple linear score fusion of neural cosine improves over our 38-feature GBDT ensemble.
2. **Phase Progression:** If $\Delta F_{0.5} > 0$ with positive CI, integrate neural cosine into the production bundle. If $\Delta F_{0.5} \le 0$, proceed cleanly to **Phase R3 (Targeted False Positive / Miss Mining)** and **Phase R4 (Deeper India K250)** as prescribed by the plan.
3. Execution completed in 1132.0s.

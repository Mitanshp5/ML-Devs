# Amazon ML Challenge: Business Entity Resolution Progress Log

## Current Status — 27 September 2026, Post-Execution of `NEXT_IMPROVEMENT_PLAN.md` (Commit `bdccb1b`)

**Active Champion Architecture:** N06 Calibrated Model Ensemble ($0.60 \times \text{Matcher\_25k} + 0.40 \times \text{Matcher\_12k}$) with `FEATURES_V3` (38 features) and calibrated `country_dual` routing.
- **Full 15,000 Comparison Benchmark Macro $F_{0.5}$:** **`0.915536`** (Fine Global) / **`0.914891`** (Country Dual) — strictly positive improvement ($\Delta = \mathbf{+0.006063}$ over clean B0 baseline, 95% bootstrap CI: `[+0.004222, +0.007913]`; $\Delta = \mathbf{+0.001990}$ over L04 12k rich matcher, 95% bootstrap CI: `[+0.000774, +0.003207]`).
- **Unexposed 13,000 Subset Macro $F_{0.5}$:** **`0.915855`** (outside screen_2k).
- **Screen 2,000 Subset Macro $F_{0.5}$:** **`0.913460`**.
- **Fresh Locked Holdout (1,000 queries):** **`0.913900`** with **96.69% Pair Precision** (1,248 true positives, 43 false positives).
- **Official Submission Validator:** **`PASS — no blocking issues found. Safe to submit.`** (Exit code 0 on all 1,732,544 rows).

---

## 1. Verified Improvement Progression (N00–N07)

| Milestone | Phase | Description | Population | Macro $F_{0.5}$ | $\Delta$ vs B0 Baseline (95% CI) | Status & Artifacts |
|---|---|---|---|---|---|---|
| **B0** | Baseline | Clean Lexical + Structured Reference (23 feat) | screen_2k | 0.904586 | *Reference* | `runs/parallel-v1/d1/b0_baseline/` |
| **B0** | Baseline | Clean Lexical + Structured Reference (23 feat) | comparison_15k | 0.909473 | *Reference* | Baseline 0.70 policy |
| **L04** | Feature Engineering | 38 Rich Features (Decomposed Channels + Competition Margins + Conflict Flags) | screen_2k | 0.912044 | +0.007459 [+0.002362, +0.012808] | [`reports/dev_probe/L04_richer_features_report.md`](reports/dev_probe/L04_richer_features_report.md) |
| **N00** | Repair | Resolved G01 empty-name defect; registered 1k holdout ledger; restored L04 runner | — | — | — | Unified `serialize_record_text()` in `normalization.py` |
| **N01** | Confirmation | 15k Benchmark Confirmation & Loss Taxonomy | comparison_15k | 0.913546 | +0.004072 [+0.002158, +0.006052] | [`reports/dev_probe/N01_comparison_confirmation_report.md`](reports/dev_probe/N01_comparison_confirmation_report.md) |
| **N01** | Confirmation | Unexposed 13k subset outside screen | unexposed_13k | 0.913777 | +0.003551 [+0.001443, +0.005637] | 55.7% perfect queries; 34.4% false negatives; 6.2% false positives |
| **N03** | Identity Scaling | Scaled to 25k Training Identities (2.5M pairs, 438 trees, 3-fold inner GroupKFold) | comparison_15k | 0.914325 | +0.004851 [+0.002980, +0.006819] | India lifted to **0.892296** (+0.256 pp); [`reports/dev_probe/N03_identity_scaling_report.md`](reports/dev_probe/N03_identity_scaling_report.md) |
| **N06** | Ensembling | Calibrated Model Blend ($0.60 \times 25k + 0.40 \times 12k$) strictly selected on `calibration_5k` | comparison_15k | **0.915536** | **+0.006063** [+0.004222, +0.007913] | **Active Champion**; unexposed 13k reaches **0.915855**; [`reports/dev_probe/N06_ensemble_report.md`](reports/dev_probe/N06_ensemble_report.md) |
| **A01a**| Semantic Smoke | Multilingual MiniLM throughput benchmark & diverse embedding verification | 1,000 samples | 245.6 texts/s | Off-diagonal cosine std = 0.206 | Non-constant cosine verified; [`reports/dev_probe/A01a_neural_benchmark_report.md`](reports/dev_probe/A01a_neural_benchmark_report.md) |
| **N07** | Holdout Assessment| Single-pass assessment on 1,000 fresh, untouched holdout queries (500 IN / 500 US) | holdout_fresh | **0.913900** | +0.001639 vs L04 Base | Pair precision: **96.69%**; registered in `exposed_holdout_2k.json` |
| **Prod**| Final Packaging | Production runner & official validator integration | 1,732,544 rows | **PASS (code 0)** | Zero blocking issues | [`output/matching_results.tsv`](output/matching_results.tsv), [`output/candidate_pairs.tsv`](output/candidate_pairs.tsv) |

---

## 2. Production Bundle Specification ([`production_bundle/`](production_bundle/))

- **Primary Scaled Matcher (25k identities):** `production_matcher_25k.txt` (SHA-256: `3b9f3322c287954bd97cdcf454004b8ae3ce50e9161e7a3414407dda26a36f06`, 3.09 MB)
- **Secondary Rich Matcher (12k identities):** `production_matcher.txt` (SHA-256: `62a2dba2388e8089daf84e17e86e6cdc0f3bc20fbc5ea2eace18870e80e6039c`, 6.38 MB)
- **Feature Schema:** `feature_schema.json` (`FEATURES_V3`, 38 features)
- **Calibrated Decision Policy:** `decision_policy.json`
  - Ensemble Weights: $0.60 \times \text{Matcher\_25k} + 0.40 \times \text{Matcher\_12k}$
  - India: $t_s = 0.66, t_m = 0.63$
  - US: $t_s = 0.61, t_m = 0.61$
  - France / Global Fallback: $t_s = 0.64, t_m = 0.63$
- **Production Manifest:** `production_manifest.json` with cryptographic provenance and verified metrics.

---

## 3. Official Validator & Code Verification

- **Official Validator:** Executed `student_resource/student_resource/utils/validate_submission.py` against output TSVs. Result: `PASS — no blocking issues found. Safe to submit.`
- **Pytest Suite:** 20/20 unit tests pass (100%).
- **Knowledge Graph:** 886 nodes, 1,206 edges, 132 communities via `graphify update .`.

# Amazon ML Challenge: Business Entity Resolution Progress Log

## Current status — 27 September 2026, Autonomous Execution of NEXT_IMPROVEMENT_PLAN.md

**Active Champion Architecture:** N06 Calibrated Model Ensemble ($0.60 \times \text{Matcher\_25k} + 0.40 \times \text{Matcher\_12k}$) with `FEATURES_V3` (38 features) and calibrated `country_dual` routing.
**Full 15k Comparison Macro $F_{0.5}$:** **`0.915536`** (Fine Global) / **`0.914891`** (Country Dual) — strictly positive improvement ($\Delta = \mathbf{+0.006063}$ over clean B0 baseline, 95% bootstrap CI: `[+0.004222, +0.007913]`; $\Delta = \mathbf{+0.001990}$ over L04 12k rich matcher, 95% bootstrap CI: `[+0.000774, +0.003207]`).
**Unexposed 13k Macro $F_{0.5}$:** **`0.915855`**.
**Fresh Untouched Holdout (1,000 queries):** **`0.913900`** with **96.69% Pair Precision**.
**Official Submission Validator:** **`PASS — no blocking issues found. Safe to submit.`** (Exit code 0).

---

## 1. Verified Improvement Progression

| Phase | Milestone / Model | Population | Macro $F_{0.5}$ | $\Delta$ vs Clean B0 (95% CI) | Status | Key Artifacts & Notes |
|---|---|---|---|---|---|---|
| **B0** | Clean Reference (23 feat) | screen_2k | 0.904586 | *Reference* | **COMPLETE** | `runs/parallel-v1/d1/b0_baseline/` |
| **B0** | Clean Reference (23 feat) | comparison_15k | 0.909473 | *Reference* | **COMPLETE** | Baseline 0.70 policy |
| **L04** | Rich Matcher (38 feat, 12k train) | screen_2k | 0.912044 | +0.007459 [+0.002362, +0.012808] | **COMPLETE** | Decomposed channels, competition margins, conflict flags |
| **N00** | Foundation & Provenance Repair | — | — | — | **COMPLETE** | Fixed G01 empty serialization; registered 1k holdout ledger; restored L04 runner |
| **N01** | 15k Benchmark Confirmation | comparison_15k | 0.913546 | +0.004072 [+0.002158, +0.006052] | **CONFIRMED** | L04 confirmed superior on 15k and unexposed 13k (`0.913777`) |
| **N03** | Identity Scaling (25k train) | comparison_15k | 0.914325 | +0.004851 [+0.002980, +0.006819] | **COMPLETE** | Scaled to 25k identities (2.5M pairs); India lifted to 0.892296 |
| **N06** | Calibrated Model Ensemble | comparison_15k | **0.915536** | **+0.006063** [+0.004222, +0.007913] | **CHAMPION** | Blend ($0.60 \times 25k + 0.40 \times 12k$); unexposed 13k reaches **0.915855** |
| **A01a**| Multilingual MiniLM Benchmark | 1,000 samples | 245.6 texts/s | Non-constant cosine (std=0.206) | **BENCHMARKED**| Repaired serializer verified; full CPU encoding extrapolated to 41.2 min |
| **N07** | Fresh Locked Holdout | 1,000 holdout | **0.913900** | +0.001639 vs L04 Base | **VERIFIED** | Evaluated on fresh, untouched holdout identities; 96.69% precision |
| **Prod**| Submission Pipeline & Validator | 1,732,544 rows| **PASS (code 0)** | Official validator certified | **VERIFIED** | Formatted `matching_results.tsv` and `candidate_pairs.tsv` |

---

## 2. Production Bundle State (`production_bundle/`)
- `production_matcher_25k.txt`: SHA-256 `3b9f3322c287954bd97cdcf454004b8ae3ce50e9161e7a3414407dda26a36f06`
- `production_matcher.txt`: SHA-256 `62a2dba2388e8089daf84e17e86e6cdc0f3bc20fbc5ea2eace18870e80e6039c`
- `feature_schema.json`: SHA-256 `c5c12c96da534eb343daa2b5b2732e82ee88488b40482901601af687d3c5810c`
- `decision_policy.json`: SHA-256 `bbf6bd1cb761c381e00fa8db960cccc05b949a0fab5ac24dd966c1c62be4529a`
- `production_manifest.json`: Full specification, ensemble weights ($0.60 \times 25k + 0.40 \times 12k$), routing policy (India, US, and France global fallback), and verified metrics.

---

## 3. Test Suite & Invariant Verification
- **Pytest:** 20/20 unit tests pass (100%).
- **Git Invariants:** Zero untracked or tracked files > 45MB; clean Git LFS compliance.
- **Knowledge Graph:** Updated via `graphify update .` (867 nodes, 1159 edges, 126 communities).

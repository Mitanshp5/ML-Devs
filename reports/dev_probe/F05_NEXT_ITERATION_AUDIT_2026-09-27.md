# Further-improvement audit — 27 September 2026

Repository reviewed at `7875f57`. This audit checks the pasted completion summary against L01–L07/G01 reports, current source, cached predictions, text records and production artifacts. It performs no training and no new holdout evaluation. The implementation queue is [NEXT_IMPROVEMENT_PLAN.md](../../NEXT_IMPROVEMENT_PLAN.md).

## 1. Verdict

There are worthwhile experiments left. The 38-feature matcher improves the existing screening result, but has not yet been confirmed on the larger 15k development comparison. The neural experiment has a verified input-schema bug and does not establish whether multilingual features help. Increasing training identities and retraining on deeper candidates remain untested routes.

Do not interpret the reported 0.924080 holdout score as a +0.019494 improvement over B0's 0.904586 screen score: those are different query populations, and B0 was not evaluated alongside L04 in that holdout report.

## 2. Recomputed L04 screening policies

Recomputed directly from `runs/local-v2/L04_richer_features/screen_predictions.parquet`, the complete B0 screening truth, cached query countries, and L04's saved `calibrated_policies.json`, using `prepare` and `apply_policy` from `er.analyze_b0_decisions`.

| Saved policy | Overall macro F0.5 | India | US | False-positive pairs | False-negative pairs |
|---|---:|---:|---:|---:|---:|
| baseline: fixed 0.70/0.70 | 0.912044413 | 0.884169221 | 0.939919605 | 214 | 1,077 |
| fine_global: 0.655/0.655 | 0.911343419 | 0.882290060 | 0.940396779 | 249 | 1,007 |
| fine_dual: same thresholds here | 0.911343419 | 0.882290060 | 0.940396779 | 249 | 1,007 |
| country_dual | 0.910986384 | 0.882290060 | 0.939682708 | 247 | 1,023 |

The 0.912044 figure is reproducible, but belongs to the fixed 0.70 policy. The L04 Markdown/JSON report describes country_dual at 0.910986. `export_final_model.py` hardcodes 0.912044 into a manifest whose routing describes country_dual. These are different systems and must be named separately.

The manifest also describes France fallback as 0.70/0.70; the saved country_dual policy has global fallback 0.655/0.655. An explicit selected-policy identifier and machine-generated metrics are needed.

All three production artifact hashes match the manifest. The production model has 466 trees and 38 features, and its feature names exactly equal `FEATURES_V3`. Integrity of these files does not resolve the policy ambiguity or establish full-test completion.

## 3. Verified neural input bug

`code/business_entity_resolution/src/er/training/run_g01_local_multilingual.py::serialize` reads `business_name` and `business_address`. The B0 text bundles actually store `name`, `address`, and `country`.

Consequently, this runner serializes the inspected bundle records as ` |  | India` or ` |  | US`. Candidates are country-partitioned, so this gives identical query and target text within each country.

Direct cache checks:

- Calibration: 500,000 rows; `neural_cosine` has one unique value, exactly 1.0, for both positive and negative pairs.
- Screen: 200,000 rows; the same constant 1.0 result.
- First target embedding shard: 50,000 vectors of 384 float32 values; only two distinct vectors among the first 1,000 inspected entries.

This explains the small embedding files and invalidates the claim that this experiment demonstrates useful neural representations cannot help. It tested a constant feature. The report remains a historical execution record, not a valid neural-quality comparison.

A second issue remains after fixing serialization: the fusion sweep changes the score scale but keeps B0 thresholds fixed. Any corrected fusion arm must recalibrate its own policy on calibration data. Raw cosine is not a match probability. The current false-negative recovery diagnostic also uses a fixed 0.70 cutoff rather than the actual country policy.

Do not reuse these embedding or neural-score caches after correcting serialization. Use a new cache namespace with hashes of the serializer, texts and model configuration; retain the original artifacts as invalidated evidence.

## 4. Current error budget

Computed from the same 2k screening predictions under the fixed 0.70 policy. This is development diagnosis, not an independent validation result.

| Query category | All 2k | India 1k | US 1k | Contribution to overall macro loss |
|---|---:|---:|---:|---:|
| Perfect prediction | 1,107 | 477 | 630 | 0 |
| False positives only | 123 | 82 | 41 | 0.017612334 |
| False negatives only | 693 | 390 | 303 | 0.054963534 |
| Both error types | 77 | 51 | 26 | 0.015379719 |

There are 9 singleton false merges and 37 false-empty non-singleton predictions. Across all pairs there are 214 false positives and 1,077 false negatives. Macro scoring weights queries, so pair counts alone do not determine the best intervention.

The natural K100 oracle remains 0.985159345. Therefore:

- Retrieval loss: `1 - oracle = 0.014840655`.
- Downstream loss: `oracle - actual = 0.073114932`.
- Total loss: `1 - actual = 0.087955587`.
- Downstream matching/decision loss is approximately **83.1%** of the current total.

India's actual/oracle are 0.884169221/0.974535802; US actual/oracle are 0.939919605/0.995782888. Prioritize richer matching and missed-positive analysis, with selective India retrieval work. Do not merely lower all thresholds: the recomputed table already shows that more recovered pairs can reduce macro F0.5.

## 5. What the other reports establish

| Experiment | Supported conclusion | Limit |
|---|---|---|
| L01/L02 tree capacity | B0 country_dual scores 0.910485 on full 15k; capacity_127 country_dual scores 0.909067 | More tree capacity did not confirm the screening gain |
| L02 country policy | Full-15k delta +0.001012, CI [+0.000073, +0.002048] | On the 13k outside screen, delta +0.000831, CI **[-0.000244, +0.001926]**; not strictly positive |
| L03 depth, frozen B0 | K100 actual 0.906774; India K250 0.906710; all-untrimmed 0.906836 | Oracle improves, actual gains unresolved; no corresponding rich-model/deeper-training experiment |
| L04 rich features | Country_dual 0.910986, delta vs B0 fixed policy +0.006401, CI [+0.001217, +0.011706] | Delta vs B0 country_dual +0.004212 has CI [-0.001002, +0.009289]; no L04 15k report |
| L05 India ablation | Address channel is valuable; dropping it reduces oracle to 0.930313 | Address K300 at final K100 worsens oracle to 0.973702; address K300/final K250 reaches 0.980223, already reached by L03 K250 at reported precision |
| G01 neural fusion | A constant feature receives weight zero | Does not test informative multilingual embeddings |
| L07 holdout | L04 fixed policy 0.922639; fine_global 0.924080; country_dual 0.922848 on 1k | These policies share the L04 model; the row called baseline is not the old B0 model |

L05's `oracle_recall` column is an oracle macro-F0.5 value, not pair recall. Removing a channel also changes the capped RRF candidate selection; a slight improvement after removing name retrieval does not establish that name evidence is useless.

## 6. Reproducibility and exposure gaps

The following tracked Python files have size zero at this snapshot:

- `src/er/training/run_l01_matcher_screen.py`
- `src/er/training/run_l03_candidate_depth.py`
- `src/er/training/run_l04_richer_features.py`
- `src/er/run_calibration_sweep.py`
- `src/er/run_compression_sweep.py`

Paths above are relative to `code/business_entity_resolution/`. Restore or rebuild the required L04 training entry point before claiming a reproducible next baseline. The saved artifacts are not evidence that these current entry points can reproduce them.

`evaluate_holdout.py` samples 500 India and 500 US holdout queries with NumPy seed 42, preserving the order derived from `splits.json` and source-country filtering. It writes the score report, but does not persist selected query IDs or keyed holdout predictions. Reconstruct and save those IDs from the exact input ordering without rerunning inference or reopening labels for tuning. Mark them exposed; leave the rest of the holdout unused. If historical inputs cannot be verified, use a conservative exposure record rather than claiming an exact reconstruction.

The 13k outside the screen are now exposed development data. They are useful for consistent comparison, not a fresh blind set. Since several holdout policy results are visible, choosing a policy because it won that table would be holdout-based selection.

The production bundle is a model/schema/policy/manifest export. The inspected L07 export/evaluation scripts do not generate the complete test candidate and matching outputs or run the submission validator. Complete those operational stages separately before calling a submission finished.

## 7. Audit scope

Read the latest L01–L07/G01 reports, relevant earlier B0/audit context, source and policies; recomputed L04 screening metrics and error categories; inspected cached text schema and neural-score distributions; checked embedding variation in one shard; verified production hashes and feature names. No model was retrained, no full test run was started, and no additional holdout performance was measured.

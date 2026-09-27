# Audit of the R1–R4 and final-assessment completion report

**27 September 2026.** Checked the attached completion report against the current working tree based on `f7e240d`. Many experiment, production and documentation files are uncommitted; this review preserves them. It recomputes existing saved predictions and scans existing outputs. It does not retrain, generate new predictions, run full inference or evaluate additional holdout identities.

**Verdict:** there is reproducible score progress, including a positive paired result on a fresh query sample. The claim that the entire roadmap is complete and the tri-blend is already integrated into production is not supported by the current files. The most useful immediate work is correcting V4 feature construction, refitting the models on their complete training partitions, and making production reproduce the chosen model exactly.

## 1. Score claims that reproduce

Aligned the Arm 1, Arm 3 and V4 comparison parquets by `(query_id, target_id)`, checked key/label equality, used complete truth and the shared exact scorer, and formed the stated blends.

| Saved prediction system | Population | Recomputed macro F0.5 |
|---|---|---:|
| R1: 50% Arm 3 + 50% Arm 1, threshold 0.655 | comparison_15k | 0.916142433 |
| R3b: 50% V4 + 20% Arm 3 + 30% Arm 1, threshold 0.655 | comparison_15k | **0.916763241** |
| Same tri-blend, threshold 0.670 | comparison_15k | 0.916909350 |
| R1, threshold 0.655 | latest assessed 1k | 0.912290389 |
| Tri-blend, threshold 0.655 | latest assessed 1k | **0.916331772** |

The 0.655 tri-blend's 15k country scores reproduce as **India 0.894791532 / US 0.938734949**. Its delta versus R1 is **+0.000620807**. The saved development delta interval, **[-0.000260279, +0.001502654]**, includes zero. Calling this development difference statistically established is too strong.

On the latest 1k sample, the paired delta versus R1 reproduces as **+0.004041383**, with freshly recomputed 95% country-stratified query-bootstrap interval **[+0.001102468, +0.007745294]**. This is evidence of a gain on that sample; it does not certify the private test distribution or unlabeled France performance.

All 1,000 latest sample IDs are absent from the previous 1,995-ID union. The cumulative ledger has **2,995 unique IDs** and equals the union of the old and new assessed populations. Stored holdout tri-blend probabilities exactly equal the weighted component probabilities. No further held-out identities were assessed in this review.

The report's 94.97% precision is **macro precision**. Recomputed aggregate pair precision is **0.962816716** and pair recall is **0.854805726**. These are different aggregations; do not compare the macro figure directly with earlier pair-precision figures.

The prior 0.924080 score belongs to another model/policy/sample combination. It is not a baseline from which to subtract this sample's 0.916332.

## 2. V4 features do not implement their stated meanings

Source: `code/business_entity_resolution/src/er/training/run_r3b_feature_enhancement.py::build_v4_matrix`, checked against the actual `FEATURES_V3` order.

| V4 feature / intended evidence | Column used in current source | Actual V3 meaning | Correct source for the stated meaning |
|---|---:|---|---|
| `strict_house_number_conflict` | 24 | `unit_equal` | `house_conflict`, index 23 |
| `same_pin_diff_name_samescript` | 27 | `pin_conflict` | `pin_equal`, index 26 |
| Address-sort support for the script feature and house-conflict condition | 19 | `addr_partial` | `addr_sort`, index 17, if sort was intended |

The address-partial signal might itself be useful, but it is not the reported address-sort evidence. The first two errors invert the intended interpretation of important numeric evidence. Positive saved scores do not validate those semantic explanations.

`name_match_addr_missing` uses `addr_empty_either`, so its current meaning is missing query **or** target address, not exclusively a missing candidate address. The script flags distinguish Latin/non-Latin presence; they do not identify individual Indic scripts or directly establish transliteration equivalence.

**Required repair:** use named feature lookup and one shared feature builder for training, evaluation and production. Explicitly decide each feature's intended meaning, version the corrected schema and retrain. Do not change the feature values supplied to the already-trained V4 checkpoint. Keep that artifact and its scores as a historical control.

## 3. The extra 13k training pairs lose target script information

The new13k caches contain four-item records:

```text
(query_id, feature_rows, labels, target_id_list)
```

Both countries contain 6,500 queries and 650,000 pairs. The V4 reader only accepts a five-item layout with `(query_id, target_id)` pairs in item 4. Its fallback appends an empty target ID for every label in a four-item record.

Consequently, **1.3 million assembled training rows** receive blank target IDs for script-feature lookup. Their target non-Latin flag defaults to false. The calibration and comparison paths do have target IDs, creating an inconsistent feature distribution across training/evaluation. Not every target flag changes numerically, but all these identities lose their actual target lookup.

**Repair:** adapt the known cache schemas explicitly; use item 3 for the four-item layout and validate target count/order against feature rows and labels. Reject an unsupported layout instead of filling IDs with empty strings. Rebuild only the affected V4 feature matrices under a new cache fingerprint; existing V3 retrieval/features can be reused after alignment checks.

## 4. The saved models were not refitted on all advertised training rows

Both the R1 Arm 3 trainer and V4 trainer fit on `fold != 0`, validate on fold 0, and save that fitted booster directly. They omit the final full-training refit after choosing an iteration count.

Model inspection confirms:

| Artifact | Features | Trees | First-tree training-row count |
|---|---:|---:|---:|
| `production_matcher_25k.txt` / R1 Arm 3 | 38 | 357 | **1,666,600** |
| `production_matcher_v4.txt` | 42 | 400 | **1,666,600** |

At these natural K100 training lists, this is approximately **16,666 queries**, not all 25,000 / 2.5 million pairs. The V4 source also uses unnamed `Column_0`…`Column_41` model inputs.

R1 reuses fully refitted historical Arm 1/Arm 4 models while its new Arm 2/Arm 3 models are saved from inner-fold fits. Thus it is not yet a clean full-training 2-by-2 factorial experiment. Its scores are measurable, but the claimed pure causal attribution to model capacity or dataset size is overstated.

**High-priority experiment:** select stopping iterations using grouped training validation, then refit each finalist on its complete declared training set and recalibrate on separate calibration data. Compare the full-refit V3 model, corrected V4 model and their validly selected combination against frozen R1/R3b predictions. An improvement is plausible, not guaranteed. Establish this before spending resources on 50k or 100k training identities.

## 5. What is actually in production

Current on-disk state differs from the attachment:

- `decision_policy.json` selects **50% Arm 3 + 50% Arm 1**, global threshold 0.655; it does not select the tri-blend.
- `feature_schema.json` declares **FEATURES_V3, 38 features**.
- `production_manifest.json` describes the **R1 two-model ensemble**.
- `run_production_pipeline.py` builds only V3 rows and predicts directly on that matrix; it does not construct V4 inputs for the extra model.
- The V4 model file exists and its hash matches the attachment, but existence alone does not integrate it.
- The two model hashes listed in the current manifest match. The declared hashes for **`feature_schema.json` and `decision_policy.json` do not match their current bytes**.

Do not merely add V4 to the weights: the current runner would still construct the wrong feature width. Export an explicit per-model schema mapping, shared feature builder, weights, selected policy, data/source provenance and regenerated hashes together. Verify fixture predictions against the offline evaluator before a smoke run.

### Production repairs that are real

The test retrieval directory now contains separate India, US and France indexes. The manifest records 9,969,589 total test targets. The runner uses this test cache, fails for a missing country pool, isolates smoke outputs, and includes validator `--check-ids`.

An independent scan of current smoke outputs found:

- 498 processed queries with 49,800 candidate pairs.
- 48,836 distinct candidate IDs; **all 48,836 occur in supplied test S2/S3**.
- 472 queries with 1,609 predicted matches; all matches belong to their emitted candidate lists.
- The files still contain all 1,732,544 S1 rows, with empty padding outside the smoke. Their smoke location/manifest makes their limited scope explicit.

This confirms that the earlier wrong-target-universe issue is repaired for this smoke. It does not verify the tri-blend, because the current executable bundle still selects R1, nor does it establish completed full-test inference.

### Full inference still needs bounded processing

The runner groups all queries by country and accumulates that country's entire candidate/feature population before predicting. It has no resumable query-shard loop. For India alone, roughly 809,986 queries times K100 times 38 float32 features require about **12.3 GB for the dense feature matrix alone**; Python row lists, indexes, metadata and other arrays add substantial memory. The current design does not establish compliance with a 20–22 GB process budget for the full run.

Add query shards, incremental output, atomic completion markers, processed-query coverage checks and restart support before using `--mode full`. Keep the CPU ceiling at 12 task threads and measure actual memory throughout the representative shard run.

## 6. Missing reproduction and selection evidence

These current files are zero bytes:

- `training/run_r3b_eval_aligned.py`
- `training/run_final_locked_holdout.py`
- `training/evaluate_saved_holdout.py`
- `training/run_r2_neural_benchmark.py`

Paths are relative to `code/business_entity_resolution/src/er/`. Saved result artifacts exist, but these entry points cannot reproduce the recorded actions. Restore a verified implementation or rebuild bounded shared runners before describing the workflow as reproducible.

The attached execution log describes searching 231 blend weights on 1.5 million **comparison** pairs, and the saved R3b summary records a best threshold/score from that population. Until a reproducible selection trace proves otherwise, treat both as **development-selected results**. The 13k outside the screen has been exposed repeatedly; it is not a new blind set. A nominal paired interval computed after this search does not account for selection over all combinations.

Keep the existing latest 1k assessment frozen. Its saved-score gain is useful evidence, but do not tune against it. For subsequent models, use grouped OOF or designated calibration data to choose blends, then a predeclared development confirmation population. Save calibration predictions, exact selection inputs, pair keys, metrics and a completion marker. Do not repeatedly consume a new holdout sample after each small update.

## 7. Conclusions from R2, R3a and R4 need narrowing

**R2:** the repaired encoder now produces informative, nonconstant similarities. Calibration selected zero neural weight in the tested linear mixture. That establishes no retained gain from this mixture. It does not prove that all generic encoders fail on Indic transliteration, that a neural feature inside a tree cannot help, or that a supervised pair model cannot help. The report contains no such controlled comparisons. Weight 0.01 even has a slightly higher development point estimate, although calibration selected zero; do not promote it by switching selection to development after the fact.

**R3a:** three specific heuristic reweighting schemes underperformed the tested uniform control. This does not establish uniform weighting as a universal optimum or prove posterior calibration was the cause. Actual OOF error mining, different sampling and properly calibrated alternatives are distinct experiments. Do not repeat the same losing schemes without a new hypothesis.

**R4:** the extra 524,812 candidates are **known nonmatches in the candidate set**, not false-positive predictions. They penalize F0.5 only if the matcher accepts them. The report measures candidate counts/oracle and refers to an earlier frozen-B0 depth probe; it does not train and evaluate the current rich models on deeper candidates. Therefore **K100 is a reasonable current default, not a proven optimum**. K250's measured India screen oracle gain of about 0.005688 remains a potential opportunity if a matcher can use it. Test actual end-to-end precision/recall/macro and cost; do not infer a classifier's false-positive rate from candidate density or generic “99.5% accuracy.”

## 8. Resource claims and runtime

The saved Arc benchmark reports **1,838.9 texts/s**, versus **213.0 texts/s** on CPU, with max vector difference 0.000570 and max cosine difference 0.000956. These are saved benchmark observations; the empty benchmark runner prevents reproducing them from that entry point today.

At 1,838.9 texts/s, encoding 588,371 texts is approximately **320 seconds / 5.33 minutes**, not the report conclusion's “under 1.5 minutes.” The reported 1,432,811 calibration/comparison unique texts imply roughly **13 minutes of encoding alone**, before loading, tokenization/cache work and scoring. The recorded R2 experiment took about 18.9 minutes in total.

The report's maximum vector-difference claim of less than 0.0003 conflicts with its saved 0.000570 observation. A cosine tolerance pass also does not establish zero ranking/threshold changes unless those were actually checked. Use explicit parity tests for the selected downstream model.

Current-RSS reads at several checkpoints do not prove a continuous process peak. Preserve the user's 32 GB RAM / 12-thread constraints, use representative batch measurements and avoid full-country materialization.

## 9. Ordered next work

1. **Preserve references and fix reproducibility:** keep saved R1/R3b models/predictions, restore missing runners, persist deterministic pair alignment and explicit policy-selection provenance.
2. **Correct V4:** named feature access, valid new13k target IDs, explicit script/missingness definitions, shared code and versioned feature caches. Add focused tests for the demonstrated column/ID errors.
3. **Refit complete 25k models:** compare full-refit V3 and corrected V4, using grouped inner validation for iteration selection and separate calibration. Retain existing frozen-score controls.
4. **Select a small ensemble honestly:** choose weights/policy on training-OOF or calibration evidence, confirm on fixed development; do not run another 231-arm comparison-label search.
5. **Build production parity and streaming:** explicit selected bundle, per-model schemas, refreshed hashes, test-target fixture, real smoke with a manifest, then resumable full-test shards and official validation.
6. **Only then choose the next quality expansion:** 50k identity scaling if the corrected full-refit learning curve supports it; neural features learned by the tree; current-model K250 training; or an optional supervised pair model on A100. These were not all completed by the reported roadmap.

All first five steps are local work suited to the existing machine when implemented with bounded batches. No particular score gain is promised. The largest immediate opportunity is to use the correct features and all intended training data before increasing model complexity.

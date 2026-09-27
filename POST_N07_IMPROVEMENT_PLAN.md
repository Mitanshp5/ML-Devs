# Current status and next improvement plan after N07

## Execution update — corrected V4 refit launched

The V4 feature builder has been corrected and a new versioned run is active at `runs/local-v4/R3b_corrected/` using `--n-cores 12` with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` all set to 12. The run uses the repaired target-ID cache layout, named V3 column meanings and a full-25k refit after validation-based tree selection. It is expected to take approximately 15–25 minutes locally (10–35 minute planning range). The prior V4 artifacts remain unchanged.

This tabular run does not use the Arc GPU; OpenVINO acceleration is reserved for the corrected neural encoding path, whose measured Arc throughput is about 1,839 texts/s. No score from the corrected refit is claimed until its report is written.

### Result: corrected single-model run completed

The run completed in 329 seconds and produced `runs/local-v4/R3b_corrected/matcher_v4_25k.txt` plus a summary. The repaired single V4 model scored **0.917896662** on comparison_15k, **0.918003826** on the 13k complement and **0.917200098** on screen_2k. Its paired delta versus Arm 3 is **+0.002003040**, CI **[+0.000408832, +0.003564561]**. This is a promising result, not a production promotion yet.

The run also exposed two follow-up defects: the summary records `trees: 0` after full-data refit, and its V4+Arm1 blend row is an invalid **0.111079**. Preserve the single-model result; repair prediction persistence/alignment and tree-count metadata before any ensemble comparison or bundle update. The old champion remains the deployable reference.

## Latest revision after the R3b completion report — 27 September 2026

**Current authority:** [R3B_COMPLETION_AUDIT.md](reports/dev_probe/R3B_COMPLETION_AUDIT.md), including its ordered next-work section. The earlier review below is preserved as a historical checkpoint; several production repairs and experiments have since completed.

Verified from saved predictions: tri-blend **0.916763** on comparison_15k at 0.655, and **0.916332** on the latest 1k sample. The fresh sample has zero overlap with the previous 1,995 IDs; the ledger correctly totals 2,995. Its paired gain versus R1 is **+0.004041**, CI **[+0.001102, +0.007745]**. The 15k gain versus R1 is only +0.000621 with a zero-crossing interval, and the attached log describes selecting blend weights on development.

Test indexes and valid smoke candidate IDs are now established. However, the current on-disk production policy/schema/runner still select the **two-model V3 R1 ensemble**, not the reported tri-blend; schema/policy hashes are stale. The full runner still materializes all candidates/features for a country and requires resumable query sharding before large-scale inference.

**Immediate quality opportunity:** V4 reads `unit_equal` where it claims house conflict and `pin_conflict` where it claims postcode equality; it also discards target IDs for 1.3M added-training rows. Arm 3 and V4 were saved from inner-fold fits on 1,666,600 rows, with no final full-25k refit. Fix these observed defects before further scaling.

**Next local sequence:** restore runnable evaluation entry points → correct/version shared V4 features and cache readers → refit complete 25k V3/V4 models → select a bounded ensemble on proper calibration/OOF evidence → prove production parity and stream full inference. Then consider 50k, learned neural features or actual matched-depth training. Keep optional A100 for supervised neural training; preserve 32 GB RAM and at most 12 task CPU threads.

The R2 linear blend and three R3a weighting schemes are measured negative results with limited scope. R4's added candidate nonmatches are not predicted false positives and do not prove K100 optimal. Current K100 remains a practical default pending an actual deeper-training comparison. Do not tune again on the latest assessed holdout.

## Historical post-N07 review before R1–R4

**Reviewed 27 September 2026 at commit `f7e240d`, with the existing uncommitted `progress.md` edit preserved.** This is the active execution queue. It supersedes the status and ordering in [NEXT_IMPROVEMENT_PLAN.md](NEXT_IMPROVEMENT_PLAN.md), while retaining that document's detailed experiment designs as references.

**Hardware:** Intel Core Ultra 9 285H, 32 GB system RAM, Intel Arc 140T, OpenVINO available, maximum 12 CPU threads for the task. Required work stays local. Colab A100 remains an optional accelerator for supervised neural training.

This review recomputed development scores from saved predictions and inspected output files, source, model parameters, split manifests and exposure ledgers. It did not train a model, change a production artifact, regenerate submissions, or evaluate more holdout queries.

## 1. Current verdict

There is verified progress, but it is incremental. The richer model's improvement now survives the 15k comparison. The ensemble improves further. Corrected neural matching, hard-example training, controlled identity scaling and rich-model training at deeper candidate depth remain available experiments.

**The current submission files are not usable as completed test inference.** Fixing their target-pool and coverage defects has priority over another long model sweep. These production defects do not invalidate the separate natural-candidate development comparisons.

### Reproduced development results

All rows below refer to the same `comparison_15k` population, with complete truth and natural K100 candidates.

| Model / policy | Macro F0.5 | Interpretation |
|---|---:|---|
| B0, fixed 0.70 | 0.909473 | Original larger-development reference |
| L04 12k rich model, fixed 0.70 | 0.913546 | Confirmed feature improvement |
| L04, calibrated global | 0.913967 | Stronger calibrated L04 comparator |
| N03 25k model, country_dual | 0.914325 | Different training size **and** model settings |
| N06 60% N03 + 40% L04, country_dual | 0.914891 | Country policy result |
| N06, calibrated global 0.64 | **0.915536** | Best observed saved N06 development policy |

The ensemble global policy improves over L04 global by **0.001568925**, or **0.157 percentage points**. A freshly recomputed paired, country-stratified query bootstrap gives a 95% interval of **[0.000297372, 0.002826883]**. This is a development comparison after repeated experimentation, not a private-leaderboard forecast. The larger +0.001990 previously reported compares against L04's fixed 0.70 policy.

Verified all 1.5 million comparison pair keys and labels agree across L04, N03 and the saved ensemble. The ensemble probabilities equal `0.6*N03 + 0.4*L04` exactly in the loaded artifacts. Training_25k contains training_12k, has no query overlap with calibration_5k or comparison_15k, and all its query IDs belong to the training partition. Identity-group/fold validation remains part of the next controlled experiment.

### Holdout results must remain separate

The previous L07 sample produced 0.924080 for L04's global policy. On N07's different 1k sample, L04 fixed/global/country policies score 0.912261/0.911472/0.912421, and N06 fixed/global/country policies score 0.913900/0.913169/0.913176.

The N06 fixed-policy paired delta versus L04 fixed is approximately +0.001639, with CI **[-0.001907, +0.005790]**. N06 global and country-policy intervals also include zero. The newer sample therefore does not establish a precise holdout gain or a regression from the earlier 0.924080. Its query population differs, and selecting whichever policy wins the holdout would contaminate selection.

N07 was also not completely fresh: five of its IDs overlap L07. See the exposure repair below. Stop using these assessed samples for model selection.

## 2. Production blockers verified from current artifacts

### P0-A: wrong target universe

Directly scanned `output/candidate_pairs.tsv` and the supplied train/test S2 and S3 ID columns:

- 40,766 distinct emitted candidate IDs.
- **40,766 occur in training targets.**
- **Zero occur in test targets.**

The production runner loads `cache/retrieval/pool_dict_<country>.joblib`, the same cache family used for training/development. It substitutes the US index if a country cache is missing. That does not create a France target index.

**Repair:** build distinct, fingerprinted test S2/S3 indexes for every test country, including France. Use an explicit cache manifest containing dataset role, source hashes, country, target-ID count and normalization version. Fail if the requested country's index is absent; a generic model/threshold fallback is acceptable, a different country's target pool is not. Assert that sampled and ultimately all emitted candidate IDs belong to the correct test target universe.

### P0-B: padded smoke output presented as full inference

Both output files contain 1,732,544 rows, but:

| File | Rows with nonempty lists | Total pairs |
|---|---:|---:|
| `candidate_pairs.tsv` | **498** | 49,800 |
| `matching_results.tsv` | **5** | 5 |

The 498 candidate-bearing queries comprise 166 each from India, US and France. The runner's default mode is `smoke`; it scores the sample and then writes every test S1, filling unprocessed queries with empty lists. Empty matches can be legitimate predictions, but unprocessed queries must not masquerade as scored singletons.

**Repair:** write smoke output to an explicitly named smoke directory with a processed-query manifest. For the full run, stream all queries in resumable shards and verify actual processing coverage, not merely output row count. Do not retain all countries' feature rows or candidate dictionaries in memory. Keep processed-empty and unprocessed states distinct until every shard is complete.

### P0-C: validator and model-policy parity

The runner's validator call omits `--check-ids`. Its PASS establishes only the checks invoked, not valid target membership or completed model inference.

The current bundle contains both models, but `decision_policy.json` still contains L04-only thresholds and no ensemble weights. The runner silently supplies default 0.6/0.4 weights and then applies those old thresholds. The manifest still describes only the 12k L04 model. This is not the verified N06 pipeline.

**Repair:** export explicit model weights, a selected policy, thresholds/global fallback, both model hashes, schema and source/cache provenance. Select the policy through the development/calibration protocol; do not use the N07 winner. Check that shared development and production scoring produce identical probabilities and decisions for a fixed fixture.

Run the supplied validator with `--check-ids` after true test indexing and inference. Save its exact command/output. Also verify all-query processing, match-subset-of-candidate, candidate country, duplicate IDs and deterministic shard assembly. Preserve current files as historical smoke artifacts; do not submit them.

## 3. Remaining error budget

Recomputed on 15k for N06 global 0.64:

| Quantity | Value |
|---|---:|
| Actual macro F0.5 | 0.915535703 |
| Natural K100 oracle macro F0.5 | 0.988142530 |
| Retrieval loss: `1 - oracle` | 0.011857470 |
| Matching/decision loss: `oracle - actual` | 0.072606827 |
| Downstream share of total loss | About **86.0%** |
| Perfect queries | 8,595 |
| False negatives only | 4,736 |
| False positives only | 1,079 |
| Both error types | 590 |
| False-positive pairs / false-negative pairs | 1,808 / 7,485 |
| Singleton false merges / false-empty non-singletons | 101 / 228 |

This supports prioritizing better pair discrimination. A blanket precision increase or larger candidate cap cannot by itself address the dominant loss. Preserve precision with more informative evidence while recovering genuine rejected matches. Analyze per-query macro loss, not pair counts alone.

India remains the main country opportunity: the ensemble global result is 0.892741 in India versus 0.938330 in the US. This does not establish France quality; retain an explicitly tested generic route.

## 4. What has and has not been completed

| Stage | Verified status | Next consequence |
|---|---|---|
| N00 | Serializer repair and restored L04 runner are present; old and new exposure files exist | Do not redo the earlier missing-file repair; fix ledger interoperability and bundle parity |
| N01 | Rich model confirmed on 15k; cached predictions and error sample exist | Reuse comparison caches |
| N03 | 25k model trained on 2.5M pairs; 438 trees | Run a controlled configuration/size comparison before 50k |
| N06 | Blend predictions and policy family saved; gain reproduced | Preserve as reference; fix policy naming/selection implementation |
| A01a | Corrected serializer produces diverse fixture embeddings; CPU throughput benchmark exists | Correctness smoke is not a neural-quality experiment |
| N02 full neural features/fusion | No completed quality report or saved full neural prediction run found | Still a high-priority local experiment |
| N04 targeted feature ablations | No completed result found | Still pending |
| N05 rich matched-depth training | No completed result found | Still pending |
| A02 supervised pair model | No completed result found | Optional high-impact challenger |
| N07 | Additional 1k assessment saved as aggregate metrics; five repeated IDs | Preserve results, repair exposure ledger, do not rerun for tuning |
| Full test/package | Smoke output exists, with wrong target IDs | Incomplete; P0 repair required |

Do not interpret a commit message stating completion of N01–N07 as evidence that every optional or skipped experiment was performed.

## 5. R0 — establish the next reference and fix bookkeeping

**Local; cheap relative to retrieval/training.**

1. Freeze N06 predictions/model hashes and separately name the fixed, global, dual-global and country policies. Keep L04 global as a simple comparator.
2. In `run_n06_ensemble.py`, `best_p['global']['calibration_f05']` is currently used while the log calls it country_dual quality. Evaluate the actual chosen policy with `apply_policy` when selecting weights. The saved calibration summaries indicate weight 0.6 also wins the routed country average in this run, so this defect does not automatically change the existing blend. Repair its semantics before extending the search.
3. Use a small predeclared blend/threshold search with split or cross-fitted calibration. Do not repeatedly enlarge the search on the same 5k labels.
4. Normalize exposure ledgers. The first file stores `query_ids`; N07 looks for `exposed_holdout_query_ids`, so it excludes zero old IDs. The new file stores `fresh_holdout_query_ids` and incorrectly reports cumulative exposure as 1,000. The union of both observed lists contains **1,995 unique IDs**. Persist that union under one schema, include source-ledger hashes, and exclude the complete union thereafter. Never drop original ledgers.
5. Keep both old assessment samples frozen. The 13k outside screen is exposed development, not an unexposed/blind population.
6. Save keyed predictions and per-query metrics for every future evaluation. N07 retains aggregate JSON but not the pair predictions needed for later paired auditing; do not reopen its labels merely to repair an old report.
7. Restore accurate status reporting. The worktree `progress.md` currently differs from HEAD and shows an older pre-N01 snapshot; append an authoritative current review without discarding that existing edit.

**Exit:** reference/policy/cache IDs agree; a single cumulative exposure ledger is authoritative; report numbers are generated from saved results.

## 6. R1 — isolate training-size effects before expanding further

**Local; reuse the existing 12k and new-13k feature caches.**

The N03 comparison changed multiple variables:

| Configuration | L04 | N03 |
|---|---:|---:|
| Training queries | 12k | 25k |
| Leaves / maximum depth | 127 / 9 | 63 / 8 |
| Minimum leaf examples | 100 | 50 |
| Feature fraction | 1.0 | 0.8 |
| Bagging fraction / frequency | 1.0 / 0 | 0.8 / 0 |

With bagging frequency zero, the configured N03 row subsampling does not perform periodic bagging. N03 also selects iterations using `val_fold = 0`; it does not run three separate training/validation fits just because the fold manifest has three labels. Its logged peak-RSS helper returns current RSS, not a monitored peak.

Run a small 2-by-2 comparison: 12k/25k crossed with L04/N03 parameter settings. Reuse existing arms where their folds/configuration can be matched; otherwise perform cheap cached-feature refits. Hold candidate policy, normalization, features and calibration protocol constant. Use the same grouped validation design, record the exact folds, and evaluate every arm on the same 15k.

Only after identifying the better size/configuration combination, run three seeds for the finalist and consider 50k identities. Treat enabled bagging as its own subsequent ablation rather than changing it invisibly during the size comparison. A 100k jump is not justified by the current mixed scaling result.

**Artifacts:** exact configs, learning curves, fitted models, calibration policies, keyed predictions, paired deltas and sampled memory measurements. **Success:** larger-data benefit survives calibrated comparison and acceptable country slices, or a useful negative result prevents wasted scaling.

## 7. R2 — complete the corrected neural-feature experiment locally

**Local OpenVINO/CPU inference; no supervised transformer training required for this stage.**

A01a measured **245.6 texts/second** on 1,000 dataset queries and reported nonconstant fixture vectors. Its 607,371-text projection includes all 19k query texts in the bundle; calibration plus screen actually needs 7k queries plus 588,371 candidate targets, before any deduplication. The projection is roughly **40–41 minutes of encoding alone at that measured rate**, not a completed run or a total pipeline runtime.

The benchmark source uses `SentenceTransformer` directly. It does not demonstrate Arc/OpenVINO throughput or enforce the stated 12-thread limit by itself. Rebenchmark a representative 10k sample across country, script and text length, recording actual device and thread configuration. There is no user-imposed one-hour experiment deadline; use checkpointed runs and measured resource budgets instead of the benchmark script's arbitrary one-hour gate.

1. Validate a fresh schema/text/model-hashed embedding cache; never reuse G01's country-only vectors.
2. If exporting to OpenVINO, compare CPU and Arc embeddings/pair similarities on the same fixture within a documented tolerance. Measure downstream threshold-sensitive decisions as well as throughput. Do not change pooling, normalization or truncation silently.
3. Encode required unique calibration/screen texts in resumable shards; save keyed cosines and positive/negative/error-slice distributions.
4. Compare a small recalibrated score blend with zero neural weight, then a rich tree with the neural feature. The tree arm requires **training** embeddings too; encoding evaluation texts alone does not train a neural-feature model.
5. Reuse the valid train_12k pair/text mapping first. Add the new13k target texts only for an explicitly defined 25k neural-feature arm.
6. Confirm any screening gain on comparison_15k, with additional required texts encoded once. Retain complete truth and all query IDs, including any no-candidate queries.

**Success:** neural evidence yields a repeatable paired macro gain or a clear negative result on valid inputs. A correctness smoke or cosine separation alone is not a model-score improvement.

## 8. R3 — target the observed false matches and missed matches

**Local; use the best controlled rich configuration.**

Build a training-only grouped-OOF error table. Mine hard negatives from actual false accepts, retain ordinary negatives and all naturally retrieved positives, and keep singleton queries. Do not mine using holdout or comparison labels. Weight by query as appropriate; at fixed K100, uniform per-query total weights are nearly constant and unlikely to matter alone.

Run two separately named experiments before combining:

- **Hard-example training:** bounded weights for ambiguous same-name/different-address and same-address/different-business nonmatches, plus difficult true variants. Never treat a name/address mismatch as an automatic negative without supplied labels.
- **Feature repair/addition:** correct empty-string similarity behavior under a new schema; add train-derived rare-token agreement and more precise numeric-conflict/missingness evidence where local error inspection supports it.

Track transitions from false negative to true positive and true positive to false negative, singleton false merges, total per-query macro change and country slices. Higher pair precision with a lower macro F0.5 is a failed promotion. The known error budget justifies both recovering genuine rejected matches and reducing false positives.

## 9. R4 — deeper India training, after discrimination improves

Compare India K100 versus K250 with US K100 fixed. Generate natural training candidates at the corresponding depth, rebuild rank/competition features, refit and recalibrate. Existing frozen-B0 depth results do not answer this experiment.

Measure new true targets recovered, newly accepted false targets, actual macro delta and cost. Only extend to an untrimmed union or selective expansion if this comparison helps. Do not expand address retrieval to K300 by default; its earlier capped probe did not establish a unique benefit.

## 10. R5 — optional supervised pair-model experiment

For a substantial architectural gain, retain the [joint pair-model design](NEXT_IMPROVEMENT_PLAN.md#11-priority-revision-pursue-a-substantial-gain): field-marked multilingual records, supplied train labels, natural positives, difficult negatives, and a structured/neural combination trained without in-sample score leakage.

This stage is independent of whether a simple cosine blend helps. Pilot a compact pair classifier for 1–3 epochs using grouped training validation. Score a bounded full-K100 development set first to measure model potential, then evaluate selective inference that preserves the tree's handling of unrouted pairs. Honor multiple-match and singleton semantics.

Run a short local CPU training benchmark only if pursuing local fine-tuning; OpenVINO inference optimization is not evidence of accelerated training. Use A100 optionally for training if measured CPU cost is excessive, and test export/parity for local inference afterward. Do not make the working tree pipeline depend on this experiment succeeding. No new checkpoint/model-license choice is prescribed by this review; apply the challenge's eligibility constraints before selecting one.

If a pair model improves quality but inference is costly, measure selective routing or a smaller distilled student trained only on permitted training identities. A 20-candidate route over every test query entails approximately 34.65 million pair evaluations; measure throughput before committing to full-test neural inference.

## 11. Local schedule, memory and runtime

Use one memory-heavy retrieval/extraction/training job at a time. Enforce at most 12 task CPU threads across LightGBM, Torch, BLAS/OpenMP, tokenization and Python workers. Set library thread controls explicitly and record actual settings. This is not proof of CPU-core affinity. Arc allocations compete for shared system memory; do not treat an adapter memory label as additional dedicated RAM.

Start with 500–1,000-query feature shards and adjust using measured peak memory. Keep several GB available for Windows, caches and shared-GPU work; target roughly 20–22 GB peak task RSS until representative measurements justify a change. Monitor during extraction and training, not just at the end. Free a country's indexes before loading another when necessary.

| Work | Existing evidence | Next-run expectation |
|---|---|---|
| N01 15k extraction/comparison | 51.41 minutes | Reuse cache; do not pay this cost for every model |
| N03 new13k extraction and 25k fit | 46.94 minutes; reported RSS about 13.9 GB | Existing rows make the controlled refits cheaper; true peak still needs monitoring |
| N06 blend/policy search | 1.69 minutes | Cheap, but bound repeated calibration search |
| Corrected text inference | 245.6 texts/s on a small CPU sample | Approximately 40–41 min for calibration/screen encoding alone at that rate; train/15k coverage and Arc throughput unmeasured |
| Test-index creation/full inference | Not completed by the current smoke | Pilot each real test-country pool and extrapolate by query/shard count; no credible total yet |
| Supervised neural fine-tuning | No local/A100 run measured | Benchmark before promising hours or days |

**First session:** R0 reference/ledger repair, production test-index and policy-parity smoke, then the R1 cached-feature controlled comparison. Start corrected neural inference after measuring memory and explicit device settings. Progress to R3/R4 or optional R5 based on observed complementarity, not a predetermined promise of a large gain.

After choosing a finalist, freeze the full pipeline and conduct a single planned assessment on still-unused holdout identities if needed. Avoid consuming another 1k sample after every minor change. Save keyed results. Full-test completion requires actual scored-query coverage, real test targets and the specified validator checks; only then create the final reproducible package.

## 12. Review evidence and limits

Inputs inspected: N01/N03/N06/A01a/N07 reports and JSON, saved 15k prediction parquets, model files, production bundle and runner, training/ensemble/holdout source, split manifests, both exposure ledgers, both output TSVs, and complete train/test target-ID columns. The review recomputed development metrics/uncertainty and output membership, rather than accepting completion statements from `progress.md`.

The current task delivers this plan and status review. It does not silently replace the champion, correct source code, run a long experiment or overwrite the historical outputs. Results establish a small development gain and substantial remaining work; they do not support a promised final score.

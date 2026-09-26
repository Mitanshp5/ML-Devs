# Next iteration: improve verified entity-macro F0.5

**Updated 27 September 2026, against commit `7875f57`.** This is the active next-step plan, superseding the completed or outdated queue in `LOCAL_COLAB_IMPLEMENTATION_PLAN.md`. All required work stays on this Windows device. Optional Google Colab A100 work can run alongside it. The objective is higher reliable F0.5 at practical cost, with no fixed 0.98 target.

This document specifies work to implement and execute next. The audit itself did not launch training, Colab, full-test inference or another holdout evaluation. Detailed supporting evidence is in [the new audit](reports/dev_probe/F05_NEXT_ITERATION_AUDIT_2026-09-27.md). Retain the challenge constraints and submission requirements in [the original implementation plan](F05_098_IMPLEMENTATION_PLAN.md).

## 1. Decision and priorities

**Yes, further improvement is plausible.** There is an observed rich-feature screening gain and substantial remaining matching loss. There is also a concrete neural-input defect to fix before drawing conclusions about neural methods. None of this supports a numerical forecast for the private leaderboard.

Recommended order:

1. Reconcile model/policy provenance, restore runnable rich-model training, and register the already-exposed holdout sample.
2. Confirm L04 on the existing 15k development comparison and produce a richer error report.
3. In parallel, repair the neural serializer and benchmark valid embeddings. For the user's requested substantial improvement, prioritize the supervised A02 pair-model pilot on A100 once input and evaluation parity pass; it need not wait for a frozen-cosine blend to succeed.
4. Increase training identities with the rich feature schema, then test query-aware training weights and complementary features in a bounded sequence.
5. Retrain a challenger on deeper India candidates; judge actual macro F0.5, not the oracle alone.
6. Integrate the task-trained pair model with structured evidence, evaluate a small calibrated ensemble, and measure whether selective reranking or distillation can retain its gains at full-test cost.
7. Freeze the selected pipeline, assess it on still-unused holdout only at the end, and finish the complete test/validator/package path.

Do not repeat all completed retrieval fits or search many tree configurations first. Reuse verified candidate and feature caches where their schema/provenance match.

### 1.1 Priority revision: pursue a substantial gain

The user now explicitly prioritizes a substantial F0.5 increase. The principal architectural experiment becomes **supervised pair matching plus structured evidence**, alongside identity scaling. Corrected frozen embeddings remain a cheap diagnostic, but success of a cosine blend is not a prerequisite for a supervised pair-model pilot. A failed cosine experiment does not establish that joint pair classification will fail.

Working-tree checkpoint when making this revision: `run_l04_richer_features.py` has been restored to a nonempty implementation; `run_n02_repaired_neural.py` and a shared serializer are present as work in progress; an exposed-holdout ID file is present; `run_n01_comparison_confirmation.py` is currently empty. These are implementation observations, not verified experiment completions. No new N01/N02 quality report was found. Preserve ongoing source edits and finish their validation instead of automatically restoring or replacing them again.

Proposed scoring architecture:

```text
Country-partitioned lexical + structured candidates
    -> richer tree scores and number/address evidence
    -> supervised multilingual pair-model scores
    -> OOF-trained or separately trained fusion
    -> calibrated per-query match-set decisions
```

Start with a matched K100 population to isolate the model change. Expand India retrieval in a separate arm. The pair model reads both serialized records together and learns an identity decision using the supplied labels. This has direct precedent in [Ditto's entity-matching research](https://arxiv.org/abs/2004.00584); that paper's benchmark gains are not predictions for this challenge. The official [Sentence Transformers cross-encoder documentation](https://www.sbert.net/examples/cross_encoder/applications/README.html) describes joint pair scoring and its cost relative to separately encoded text.

| Workstream | First experiment | Conditional expansion | Execution |
|---|---|---|---|
| More independent training identities | Rich tree at 25k, then 50k, against the 12k reference | 100k only if the learning curve continues and memory/runtime are acceptable | Local |
| Supervised multilingual pair model | Existing train_12k natural positives and bounded hard/easy negatives; 1–3 epoch pilot | Extend the retained recipe to 25k/50k identities before comparing many backbones | Optional A100 |
| Structured/neural fusion | Pair-model score plus tree/number/address evidence; train combiner independently of its evaluation rows | More expressive fusion only after a simple arm confirms complementarity | Local, with A100 scoring |
| Deeper candidate training | India K250 / US K100, natural training candidates at matching depth | Untrimmed or selective expansion if actual macro improves | Local |
| Query decisions | Train a small singleton/ambiguity gate on grouped OOF training predictions | Match-count or expected-utility selection only after probability/gate validation | Local |

#### Pair-model data and objective

Use field-marked name, address and country for each record, keeping original Unicode and informative numbers. Benchmark a compact multilingual encoder before increasing size; check the exact checkpoint license/revision and the challenge's 8B-parameter limit. There is no need to start with an 8B generative model.

Build the pilot from **all naturally retrieved training positives**, including low-ranked positives, plus an initial budget of roughly four hard and two ordinary negatives per query where available. These counts are starting settings, not inferred optima. Hard negatives should include false matches with similar names but different locations, shared addresses but different businesses, short/generic names and ambiguous numeric evidence. Obtain hardness from grouped training-fold predictions or label-blind retrieval, excluding all known same-identity targets from negative sampling. Include singleton queries. Retain pair keys, identity groups, sampling probabilities and the unsampled candidate manifests.

Start with binary match classification. Evaluate the actual macro F0.5 on complete natural candidate sets after calibration; training loss or ranking quality alone is not the goal. Preserve ordinary negatives to reduce over-specialization to a narrow adversarial sample. Sampling changes score calibration, so calibrate on untouched calibration rows with the deployment candidate distribution.

Fine-tuning and binary cross-entropy are supported paths in the [official cross-encoder training guide](https://www.sbert.net/docs/cross_encoder/training_overview.html). That guide also documents a reranking evaluator setting that can automatically include unretrieved positives: use our complete-truth natural-candidate scorer, and disable any such positive insertion in auxiliary evaluators. Keep the earlier ground-truth-injection defect from reappearing through a library default.

#### Combine semantics with precision evidence

A semantic model may confuse two related businesses. Preserve exact house/unit conflicts, token rarity, missingness, source and candidate competition alongside the neural score. Compare a pair-model-only arm, structured-tree-only arm and their combination on identical candidates.

For learned fusion, generate grouped OOF scores for both supervised components, or reserve a disjoint subset of training identities for fusion and keep the base models trained outside it. Do not train a fusion layer on a base model's in-sample predictions and claim out-of-sample gains. If base models are later refit using those identities, regenerate a valid fusion-training design and calibration rather than silently retaining the old one.

Do not impose a single winner, one winner per source, a fixed predicted match count or unrestricted transitive closure. The challenge permits multiple matches. A precision gate should reject specifically ambiguous evidence while leaving supported additional matches available.

#### Establish model potential before optimizing inference cost

On a bounded development experiment, score the full existing K100 candidate lists with the pair model. This isolates how much it can improve decisions without a top-20 routing bottleneck. Then compare selective inference: top-ranked candidates plus candidates near the tree decision boundary, with the tree still handling the rest. Select routing on training/calibration and confirm its total-system quality and missed-positive coverage on development.

The previous dataset audit counted 1,732,544 test queries. Reranking 20 candidates for every query would require about **34.65 million pair evaluations**. A100 availability alone does not make that cheap. Benchmark pairs/second and full-pipeline routing volume before making the neural component mandatory for production. If a large pair model helps but is too slow, test distillation to a smaller pair model or train a compact student using teacher scores **only on allowed training identities**; measure whether the student retains the gain.

#### What size of improvement would constitute success?

The current screen diagnostic is `actual = 0.912044`, `oracle = 0.985159`, with downstream loss `0.073115`. An absolute gain of +0.02 on that same screen would require removing approximately **27%** of this downstream loss; +0.04 would require approximately **55%**, assuming retrieval stays fixed and no new errors are introduced. These calculations explain the scale of the required advance; they do not estimate its probability. They must not be added to the 0.924080 score from a different holdout population.

The first decision experiment should compare: frozen rich reference; larger-data rich tree; supervised pair model; and their validly trained fusion. Promote based on larger-development paired gains and error slices. If that comparison produces only small improvements, report that outcome instead of promising a large jump from model size alone.

## 2. Establish the correct reference

| Population | Model and policy | Macro F0.5 | Interpretation |
|---|---|---:|---|
| screen_2k | B0, fixed 0.70 | 0.904586 | Original clean reference |
| screen_2k | B0, country_dual | 0.906774 | Calibration-selected policy |
| screen_2k | L04, fixed 0.70 | 0.912044 | Recomputed; highest of these L04 screen policies |
| screen_2k | L04, country_dual | 0.910986 | Saved L04 report; different policy from the preceding row |
| comparison_15k | B0, country_dual | 0.910485 | Larger-set reference; L04 comparison is still pending |
| previously used holdout_1k | L04, fixed 0.70 | 0.922639 | Reported held-out assessment |
| previously used holdout_1k | L04, fine_global | 0.924080 | Reported assessment; do not select on this result |
| previously used holdout_1k | L04, country_dual | 0.922848 | Reported assessment of the stated production routing |

Treat the rich model as a promising challenger pending larger-development confirmation. Keep B0 and L04 artifacts read-only. Preserve all policy variants for audit, but choose one explicit policy when exporting a deployable pipeline.

The macro metric for non-singleton query truth is `5*TP / (5*TP + 4*FP + FN)`. An empty truth and empty prediction scores 1; an empty truth with any prediction scores 0. This rewards precision strongly, but does not mean recall should be ignored. Current rich-model errors include 693 queries with only false negatives, versus 123 with only false positives and 77 with both.

## 3. Evaluation protocol for this iteration

### Roles and exposure

- Keep training identities, inner folds, calibration_5k, screen_2k and comparison_15k separate using the existing manifests.
- The 2k screen is part of the 15k comparison. Report full 15k and the 13k outside the screen; neither is now blind.
- Keep the existing original train_12k fixed for feature comparisons. Expand it only in an explicitly named identity-scaling experiment.
- Select stopping iterations through grouped training folds. Calibrate thresholds using calibration data only.
- For combined fusion-weight and threshold searches, split calibration into grouped fitting/selection portions or cross-fit calibration. A small predeclared search is preferable to hundreds of combinations on the same 5k queries.
- Record all prior exposure, including the legacy 224,776 development-exposed IDs. Additional training identities must come from the training partition and respect identity grouping, never from newly convenient validation IDs.
- Register the 1k L07 holdout IDs as exposed. Do not repeatedly rerun it or use its best policy as the next tuning target. Preserve remaining holdout assignments.
- If repeated selection on the existing comparison begins to dominate decisions, reserve a new development confirmation set from unused **development** identities before the next round; do not quietly draw it from holdout. Record its exposure once used.

### Report every candidate consistently

Save keyed predictions and per-query `truth_count`, retrieved true count, predicted count, TP, FP, FN, exact score and country. Retain complete truth even when matches were not retrieved. Evaluation candidate generation must remain label-blind.

Report actual macro F0.5, oracle, retrieval/downstream loss, paired delta to the current reference, country slices, singleton false merges, false-empty queries, precision/recall, latency, peak RAM and disk. Use query-stratified paired bootstrap intervals; if multiple queries share an identity group, resample identity groups together. Verify singleton and multiple-match behavior with the existing exact scorer.

Use identical query IDs and full target pools for model comparisons. Do not describe differences across unrelated populations as improvements. Bootstrap intervals after extensive model selection are descriptive, not a cure for selection bias.

Promote a repeatable improvement that survives larger development and acceptable country/missingness slices. If an interval overlaps zero, retain the challenger as provisional, repeat only the finalist over seeds or a new development confirmation set, and prefer the cheaper model when quality remains unresolved. There is no mandatory numeric delta floor.

## 4. N00 — repair the experiment foundation

**Priority: first. Runs locally; no full-pool regeneration required.**

1. Freeze L04's saved model, prediction parquets, policies and feature caches in a manifest with source commit, feature schema and input hashes. Keep B0 as a fallback.
2. Restore the zero-byte L04 training entry point from a verified recoverable revision, or rebuild it around shared feature/training/evaluation functions. Other empty historical runners need restoration only when required by this queue. Do not blindly rerun an empty module and interpret exit code 0 as completion.
3. Add explicit CLI parameters for run directory, training-ID manifest, candidate policy, schema, seed, threads and evaluation role. Output directories must be unique; report generation must consume saved metrics rather than hardcoded values.
4. Reproduce saved L04 screen probabilities from its feature cache and model within numerical tolerance. Re-extract a bounded set of keyed pairs from natural candidate records and compare their 38 features with the saved rows, including channel ranks, numeric conflicts and missing fields. Check identity/fold and label alignment before retraining.
5. Record the training configuration represented by the existing 466-tree model. If exact historical training provenance cannot be recovered, label that limit, retain the artifact as a scoring reference, and create a reproducible successor. Do not invent a history from the report.
6. Give exported bundles a single `selected_policy` and an explicit global fallback. Correct the mismatch between the 0.912044 fixed-policy metrics, country_dual routing and France's documented threshold. Choose using development/calibration; do not resolve this by taking the best observed holdout row.
7. Reconstruct the previously evaluated holdout IDs using the original split order, country data and seed 42; persist IDs, hashes and exposure status without rerunning holdout scoring. Log uncertainty if exact inputs cannot be established.
8. Add focused checks for real observed defects: raw-vs-bundle text schemas; no silent empty serialization; nonconstant embeddings for varied fixture texts; feature/schema parity; manifest-policy consistency. Avoid duplicating implementation details in tests.

**Deliverables:** `runs/local-v3/N00_reference/manifest.json`, selected-reference metadata, reproduction report, exposure ledger and runnable rich-feature runner. Proposed paths describe future outputs.

**Exit:** cached screen metrics reproduce; selected policy and report agree; training and candidate provenance are sufficiently explicit to compare a new run.

## 5. N01 — confirm L04 on the larger comparison

**Priority: first quality experiment. Local.**

Use the existing L02 15k query IDs and natural K100 candidate population. Reuse retrieval/provenance caches if complete; enrich to `FEATURES_V3` instead of rebuilding lexical indexes. Cache the 38-feature comparison matrix once.

Predeclare two model arms, B0 and the frozen L04 model, and the small existing policy family: fixed 0.70, calibration-selected global, and calibration-selected country_dual. Preserve the saved calibrations for the first comparison. Report policies separately rather than assigning a model one selectively chosen score.

Check parity on the 2k overlap against saved predictions. Then report all 15k, the 13k complement, India/US and missingness/match-count slices with paired differences. The old capacity_127 arm gained on screening but lost on the larger set; this is why confirmation precedes scaling.

From the resulting errors, prepare a loss-ranked sample of 50–100 queries across both countries, false-positive/false-negative categories, short/common names, scripts, addresses, number conflicts, source type and truth cardinality. Classify missed positives as unretrieved, retrieved-but-rejected, or query-gated. Inspect records locally; do not look up businesses externally.

**Decision:** if rich features confirm, they become the next experiment baseline. If not, ablate channel decomposition versus competition/rank features before investing in a larger rich model. Preserve K100 as the common comparison policy until a depth arm earns promotion.

## 6. N02 / A01 — repair and properly test multilingual evidence

**Priority: high. Local smoke; optional A100 bulk encoding in parallel with N01/N03.**

The existing G01 scores are invalid for this purpose: all 700k cached cosines equal 1.0 because the serializer used absent field names. Do not load those caches into a corrected run.

### A01a: input and embedding correctness

1. Normalize both accepted record schemas into one explicit serializer. For cached text use `name`/`address`; for raw TSV records map `business_name`/`business_address` deliberately. Reject an unexpected schema. Preserve true missing fields as missing rather than inventing text.
2. Start with the existing multilingual MiniLM checkpoint so the first experiment isolates the serialization repair. Use field-marked name/address/country text; retain original Unicode. Record model revision, tokenizer, pooling, maximum length and truncation rates.
3. Test approximately 100 diverse records, including within-country unrelated businesses, same-identity variants, non-Latin text and missing addresses. Confirm text coverage, finite embeddings, intended normalization and variation across distinct inputs. Some identical records may legitimately share a vector; do not demand uniqueness per ID.
4. Cache by text hash, serializer version, model/tokenizer revision and encoding settings. Create a fresh `runs/local-v3/N02_neural_corrected/` or `runs/colab-v2/A01/` namespace. Leave invalid G01 caches intact with an invalidation note.
5. Benchmark 10k representative unique texts and extrapolate with measured throughput. The old 614-second G01 run is not a valid benchmark of encoding meaningful business records. Avoid materializing two 500k-by-384 float32 arrays for cosine calculation; process keyed pairs in chunks.

### A01b: three bounded quality arms

| Arm | Training requirement | Purpose |
|---|---|---|
| Corrected cosine diagnostic | Frozen embeddings on calibration/screen pairs | Check positive/negative and error-slice separation; no claimed model gain |
| Recalibrated small blend | Weights such as 0, 0.05, 0.10, 0.20, each with its own calibration | Cheap test; always include weight zero and preserve score-scale semantics |
| Rich tree plus neural features | Encode training candidate records too; refit on train_12k | Let the tree learn when cosine is useful alongside address/number conflicts |

For the tree arm, start with one combined-text cosine and a named 39-feature schema. Only add separate name/address cosine or margin features if the first corrected run and error report justify them. Raw cosine must not be called a probability.

Train-only identities provide supervised labels. The existing text bundles also contain evaluation query texts: their presence in the export does not authorize their use for supervised training. Exclude diagnostic unretrieved targets from natural candidate lists.

Return keyed scores, coverage, text/model hashes, runtime and embedding-health diagnostics. Confirm the best arm on the 15k comparison, which may require encoding additional target records. Do not claim confirmation from the 2k screen alone.

**Stop:** if corrected inputs and recalibration still yield no useful complementarity, retain that negative result and postpone larger encoders. This experiment tests matching existing candidates, not dense retrieval of previously missed candidates.

## 7. N03 — increase training identities, then improve training emphasis

**Priority: high after N00/N01. Local, one heavy job at a time.**

The current model still uses only 12k training identities. Test nested 12k, 25k and, conditionally, 50k training-ID sets from the allowed training partition. Maintain identity grouping and similar country/match-count coverage; log any deliberate rare-slice oversampling.

Keep the same schema, K100 retrieval, calibration set and comparison population for the first scaling experiment. Retrieve from the complete appropriate target pool, never a convenient reduced pool. Generate training candidates naturally; label them afterward. Do not add true targets to evaluation candidates.

At K100 these tiers contain approximately 1.2M, 2.5M and 5M pairs. With 38 float32 features, the raw dense matrices alone require about 182, 380 and 760 MB, respectively, before labels, IDs, Python objects and LightGBM allocations. Stream/shard extraction and measure actual peak RSS; these raw sizes are not total-memory estimates.

Use the existing rich configuration as the control. Select iterations with grouped inner folds; calibrate once per resulting model. Run 25k first. Advance to 50k only when the measured learning curve, resource cost and slices justify it. Repeat the most promising size across three seeds, rather than repeating every weak arm.

Then test training emphasis on the retained size:

- **Query normalization:** give each query comparable total training weight. At exactly K100 this is nearly constant, so do not expect benefit until candidate counts vary or sampling changes.
- **Bounded hard-negative emphasis:** select mistakes using training-fold OOF predictions, retain ordinary negatives and all naturally retrieved training positives, and cap weights so a few queries cannot dominate. No development/holdout labels in mining.
- **Bounded positive/rare-slice weighting:** try one or two modest predeclared weights; recalibrate afterward. Optimize actual macro F0.5, not training AUC or recall alone.

Only if these changes help should a country specialist be considered. Keep a pooled model as control and the global fallback for unlabeled countries. More independent identities is a higher-priority test than another broad leaves/depth sweep.

## 8. N04 — targeted feature repairs and additions

**Priority: guided by N01 error taxonomy. Local.**

Keep each feature family as a separate ablation against the same training IDs and candidate policy:

1. **Missing text:** RapidFuzz calls in V3 are not consistently guarded for empty inputs. Measure existing affected rows, then set unavailable string similarities to neutral values with explicit missingness flags. Retrain under a new schema; do not silently change feature values beneath the frozen model.
2. **Rare-token evidence:** add overlap weighted by token frequency from the supplied training target pool, plus disagreement on informative name/address tokens. Distinguish a shared generic business word from a rare token. Version the frequency table and define behavior for unseen tokens.
3. **Address/number agreement:** distinguish missing, ambiguous and confidently conflicting house/unit/postal components; compare original and normalized tokens. Inspect the parser before adding more equality flags. Avoid treating every shared number as an address match.
4. **Candidate context:** assess the count of plausible alternatives, same-name/different-address competitors, source-specific competition and score gaps. Use candidate features only, never labels. Do not impose one-match-per-query where ground truth permits multiple matches.
5. **Rank-dependence ablation:** compare V3 with rank/RRF competition features removed or reduced. Their high feature importance may be useful, but candidate-depth changes can shift their meaning. Validate across K100 and deeper candidates before deciding whether they are robust.

Do not add all families at once. Select two supported by actual loss-heavy error categories, retain their individual effects, then combine only useful ones. Corrected neural evidence is its own family under N02.

## 9. N05 — train for deeper India candidates

**Priority: after a reproducible rich baseline. Local.**

L03 only applied the old frozen B0 matcher to deeper candidates. It did not establish the effect of training the rich model on those candidates.

First compare:

| Arm | India | US | Matcher training |
|---|---|---|---|
| Control | K100 | K100 | Matched natural K100 training |
| Depth | K250 | K100 | Regenerated natural candidates at the same country policy |
| Conditional expansion, later | K100, expand uncertain queries to K250 | K100 | Train/calibrate against this staged policy |

Rebuild rank/competition features consistently. Include naturally occurring lower-ranked positives and hard negatives in training. Do not just append deeper evaluation pairs to a K100-trained model and call it a retrained depth experiment.

For each newly accepted deeper candidate, report its rank, channel, TP/FP status, and per-query macro change. Compare candidate volume, latency and RAM. The staged expansion trigger may use model confidence/margins learned from training/calibration; it cannot consult truth or simply assume every false-empty query is known in advance.

If K250 retraining helps, test the untrimmed India union. Otherwise stop increasing depth and address discrimination. L05 gives no evidence that address K300 improves the oracle over already-measured K250; do not make K300 a new default.

An optional later retrieval arm can reserve a small candidate quota for complementary channels or use diversity-aware truncation. Its value must survive actual matcher evaluation. Dropping name retrieval slightly improves the capped India probe oracle but is insufficient evidence to remove that route globally.

## 10. N06 / A02 — ensemble and task-trained reranking

### N06: cheap local combination

Once two models have complementary errors, compare one small score blend of B0 and the best rich model, or a three-seed rich ensemble. Recalibrate each arm; fit any learned stacker only on OOF training predictions or a separate training partition. Do not train a combiner on the same screen/15k results used to claim its gain.

Track which false negatives are recovered and which false positives are introduced. Two models with similar AUC or different architectures need not be complementary. Prefer the single model if the ensemble adds cost without a repeatable macro benefit.

### A02: optional A100 pair reranker

For the substantial-improvement priority in section 1.1, this pilot can follow serializer/feature parity while local identity-scaling work continues. It does not depend on a successful frozen-cosine blend. Use a compact eligible multilingual model and supplied train pairs, preserving the challenge model/license/data constraints already documented in the original plan. Start with a bounded 1–3 epoch pilot and grouped training validation; do not begin with a full-target dense index.

Construct explicit name/address pair inputs, include naturally retrieved positives and OOF-mined hard negatives, preserve numbers and measure truncation. Record negative sampling and any weighting; sampled training priors are not deployment probabilities.

First score full K100 on a bounded development experiment to measure model potential. Then test a bounded inference route, such as the top 20 plus a predeclared uncertainty band. Measure the subset's positive coverage and let the tree continue handling candidates outside the reranked subset. Evaluate the exact same routing during calibration and inference. A later top-50 arm is justified only by demonstrated missed-positive recovery; include routed training examples when fitting a specialist or fusion model.

Return keyed neural scores, model/tokenizer checkpoints, config, seed, input hashes and measured throughput. Fit/calibrate fusion locally and confirm on the same 15k population. No holdout tuning in Colab.

Dense retrieval is a separate, later experiment only if an unretrieved-positive audit shows a relevant error class and a small full-pool probe demonstrates complementary recovery. Better reranking cannot recover a target absent from every candidate list.

## 11. France, deployment and final evaluation

Retain a globally calibrated fallback selected without France labels. Inspect Unicode, accents, short names, missing addresses and number parsing on test-text fixtures. Do not manufacture a French score, tune thresholds to guessed match counts, or import external business/reference data as an assumed rules exception. Leave-country-out stress tests can reveal transfer problems but do not estimate France performance directly.

Prepare a small end-to-end production smoke early, before the last model is chosen. The pipeline must use the same normalization, retrieval, feature schema, model and policy implementation as development, cover every country, preserve singleton output semantics, and emit both required candidate and matching files. Every predicted match must be among the emitted candidates.

After development selection:

1. Freeze model, candidate policy, thresholds, global fallback, input/source hashes and production runner.
2. Preselect an adequate sample from the still-unused holdout identities. Evaluate the finalist and a predeclared reference on identical IDs in one planned assessment, so a paired improvement can actually be estimated. Save IDs, keyed predictions, metrics and uncertainty. Do not pick another threshold afterward because it wins there.
3. Distinguish any subsequent training-on-more-labels deployment refit from the assessed frozen model; it needs a documented calibration strategy and its score cannot simply inherit the frozen model's result.
4. Run full-test inference, budget candidate-file size and validator memory, verify all IDs and output invariants, run the supplied validator, and assemble the required reproducible submission package and methodology.

The current `production_bundle/` alone does not complete these stages. Keep the historical frozen output folder separate from newly generated matching/candidate pairs.

## 12. Device schedule and runtime expectations

Keep the local 12-thread ceiling across libraries, not 12 workers per concurrent process. Measure physical RAM and peak usage; the Arc GPU memory label is not a system-RAM measurement. Run one substantial local retrieval/feature-extraction/training job at a time. Small report checks can overlap if headroom permits. Colab A100 work is optional and independent; no required task moves to the Mac or RTX device.

| Stage | Local device | Optional Colab A100 |
|---|---|---|
| Foundation | N00 reproduction/provenance; A01 serialization smoke | Prepare pinned environment/export after smoke |
| First evidence | N01 rich-model 15k confirmation | A01 valid embedding benchmark/encoding |
| Main improvement | N03 25k learning curve; selected N04 feature ablations | Complete A01 train/eval features; optionally A02 pilot |
| Integration | N02 neural-feature fit; N05 matched-depth training; N06 blend | Return missing comparison scores and finalists |
| Completion | Freeze, remaining holdout assessment, full-test outputs and validator | Only necessary inference for an already-selected neural component |

Measured historical wall times from reports:

| Existing run | Wall time | Planning implication |
|---|---:|---|
| L01 cached matcher screen | 230.49 s / 3.84 min | Tree fits are cheap relative to data preparation |
| L02 larger comparison | 2,363.37 s / 39.39 min | Reuse its candidate/provenance work |
| L03 depth sweep | 1,485.23 s / 24.75 min | Expansion has measurable extraction cost |
| L04 richer feature run | 2,990.16 s / 49.84 min | One observed extraction-plus-training run; reproduction still needed |
| L05 India retrieval ablations | 908.55 s / 15.14 min | Do not rerun every channel without a new hypothesis |
| G01 faulty neural run | 614.04 s / 10.23 min | Invalid basis for valid-business-text encoding estimates |
| L07 1k assessment | 371.98 s / 6.20 min | Already completed; not a recurring tuning job |

These are observed costs of different workloads, not a promised runtime for the new queue. N01 can reuse retrieval but needs richer comparison features. N03 25k/50k costs must be estimated from a measured extraction pilot and training fit; extrapolating only row count ignores fixed indexing and memory pressure.

For encoding, use `estimated_seconds = remaining_unique_texts / measured_texts_per_second`, separately accounting for transfer, checkpoint setup and scoring. For retrieval/features, benchmark representative country shards and include index loading. For reranking, benchmark pairs per second at the chosen length/batch and multiply by routed pairs. Report a range and observed peak memory before a long run. There is no valid project-specific A100 total-runtime estimate yet.

### Recommended first work session

Complete N00, N01 and the A01a serializer/embedding correctness checks. These resolve whether the current rich model generalizes, what the main errors are, and whether there is now a real neural signal. Next run one 25k rich-model arm and one corrected neural-feature arm. Do not schedule the entire speculative queue before those results arrive.

## 13. Experiment records and stopping decisions

Each new run should write a config, input/model/schema hashes, query-role manifest, exposure status, keyed predictions, per-query metrics, policy JSON, resource measurements and a short result Markdown. Use `runs/local-v3/<experiment>/<run-id>/` or `runs/colab-v2/<experiment>/<run-id>/`; do not overwrite historical B0/L04/G01 artifacts.

Name the exact comparator and change. A useful result can be negative: it should explain which hypothesis failed and what work can be skipped. Update the champion only after matched larger-development evidence; keep uncertain arms provisional. Revisit retrieval after actual matching loss shrinks or error analysis justifies a specific new route.

Expected outcome of this plan is a reproducible, better-tested opportunity to improve F0.5. A particular score increase, private leaderboard result or competition win cannot be forecast from the available experiments.

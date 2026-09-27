# Inference optimization — 27 September 2026

The frozen V4 model, threshold 0.68, feature formulas, candidate depths, and fusion weights are unchanged. This is an execution and verification update, not a new score claim. No production inference was launched as part of this update.

## Changes

- Prepare pool matrices in CSC once per country, so each query batch receives the CSR transpose view required by sparse retrieval without converting the entire index again.
- Normalize and score at most 64 queries per feature block by default (32 on the 16 GB US device); clear the normalization cache after every block. Retrieval checkpoint batches remain at 2,000 queries.
- Offer verified memory-mapped sparse arrays for the US launcher. Arrays consume about 5.8 GiB extra disk space; dictionary/index memory remains resident. First preparation time and subsequent I/O depend on the device.
- Write shards through temporary files and publish a completion marker only after both outputs have been flushed and checked.
- Fingerprint model contents, policy, feature schema/code, input TSV, cache artifacts and numerical-library versions. Normalize text line endings for code/model fingerprints so Git checkout differences do not cause false mismatches.
- Verify query order, row/pair counts, target prefixes, subset containment and hashes on resume. Old checkpoints lacking provenance are preserved under `shards/legacy/` and recomputed. Corrupt or incompatible version-2 checkpoints cause an explicit error.
- Use OS-backed country locks to prevent two local writers from sharing the same output. Completed-country resume does not load the large indexes.
- Merge by recorded contiguous ranges, allowing different batch sizes across countries. Reject overlaps, gaps, incomplete files and fingerprint mismatches before touching existing final outputs.
- Validate both final TSVs row by row. Only source-ID sets and one candidate row are retained; the entire candidate mapping is never materialized.
- Print per-batch retrieval/total duration and a rolling provisional ETA. Save per-country manifests rather than overwriting one shared inference manifest.

## Model metadata and evaluation repair

The existing measured production model has **400 trees**, not the previously documented 394. `n_estimators=400` overrode `num_boost_round=394` during refit. The production manifest now describes the actual frozen artifact. Its model bytes and threshold have not changed. Future refits remove the conflicting parameter alias.

The ensemble evaluator now attaches sorted prediction arrays to the corresponding sorted pair table and checks the baseline pair order. No ensemble experiment was rerun. Historical invalid ensemble scores are not evidence of a poor blend, and this update makes no new F0.5 claim.

## Verification

The focused suite covers seven new optimization tests plus ten existing metric/normalization tests. The new tests cover:

1. Original versus prepared sparse retrieval candidates, scores and ranks, plus original all-at-once versus block scoring with the frozen production booster at block sizes 1, 2 and 64.
2. Resume rejection of changed model identity and output corruption; refusal to silently reuse legacy checkpoints.
3. Different country shard sizes, missing/overlapping ranges and preservation of old merged output on preflight failure.
4. Duplicate-writer exclusion and lock release.
5. RAM versus memory-mapped matrix equality and prepared-array corruption detection.
6. Streaming validation of target existence and strict match containment.
7. An entire two-query country run on synthetic caches, legacy-output preservation, and verified resume without loading indexes again.

These are small correctness fixtures, not a full-dataset speed or peak-memory benchmark. No claim of a specific speedup is justified until representative batches run. The Mac launcher was not executed on macOS in this Windows session.

## Run manually

Each device should pull the commit and install `code/business_entity_resolution/requirements-inference.txt`. Follow `DISTRIBUTED_RUN_GUIDE.md`. The India command remains:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/distributed/run_india_main.ps1
```

No job has been automatically started. Do not switch versions while another device's old inference process is still running; stop it at a completed checkpoint first, then update and rerun its launcher.

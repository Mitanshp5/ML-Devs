# Amazon ML Challenge 2026 — Final inference package

This repository contains the final, reproducible business-entity-resolution inference path for the Amazon ML Challenge. It matches every test Source 1 entity against Source 2 and Source 3 records and writes the precision-weighted F0.5 submission files.

The frozen production artifact is `production_bundle_final/production_matcher_v4.txt`: a 42-feature LightGBM matcher with threshold 0.68. Its measured development comparison score was 0.917897 macro F0.5; hidden-test performance is unknown until leaderboard scoring.

## Run inference

The test dataset and country retrieval caches are intentionally excluded from Git. See [DISTRIBUTED_RUN_GUIDE.md](DISTRIBUTED_RUN_GUIDE.md) to run India, US and France on separate devices, resume verified checkpoints, return shards, merge and validate them.

Install the inference-only environment:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements-inference.txt
```

The main-device launchers are in `scripts/distributed/`. The final leaderboard artifact is `output/final/matching_results.tsv`; `candidate_pairs.tsv` is retained for the final submission package.

## Retained source

`code/business_entity_resolution/src/er/` contains normalization, lexical/structured retrieval, duplicate expansion, features, cache building, inference, checkpoint verification and streaming validation. The original challenge statement and documentation template remain under the repository root and `student_resource/`.

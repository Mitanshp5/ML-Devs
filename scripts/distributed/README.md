# Distributed final inference

Run one country per device. Every launcher is resumable and writes independent files under `output/final/shards/`.

## Required local files

A Git pull provides the code and `production_bundle_final`, but the competition dataset and retrieval caches are intentionally excluded from Git because they total many gigabytes. Before running, copy the test dataset to `student_resource/student_resource/dataset/test/` and the relevant country cache to `cache/retrieval_test/`.

Cache sizes are approximately: France 1.31 GB, India 5.63 GB, US 3.88 GB. Copy all files containing the assigned country name plus `cache_manifest.json` when available.

Create the Python environment if the device does not already have it:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements.txt
```

On macOS use `python3 -m venv venv` and `venv/bin/python -m pip install ...`.

## Device commands

- Main 32 GB / Ultra 9: `powershell -ExecutionPolicy Bypass -File scripts/distributed/run_india_main.ps1`
- RTX 3050 / i5 HX: `powershell -ExecutionPolicy Bypass -File scripts/distributed/run_us_rtx3050.ps1`
- M4 Air: `bash scripts/distributed/run_france_m4.sh`

After France and US finish, copy their country-specific shard files into `output/final/shards/` on the main device. Do not rename them. Then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/distributed/merge_validate_main.ps1
```

The leaderboard upload is `output/final/matching_results.tsv`. The final package also needs `candidate_pairs.tsv`, the runnable code, requirements, and the filled methodology document.

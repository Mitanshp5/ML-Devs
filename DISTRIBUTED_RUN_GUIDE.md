# Distributed final-inference guide

The work is split by country. Each device creates independent checkpoints, so an interrupted run can resume without losing completed batches.

| Device | Country | Launcher | Cache download |
|---|---|---|---|
| Main 32 GB Ultra 9 | India | `run_india_main.ps1` | Already local |
| Windows i5 HX / RTX 3050 / 16 GB | US | `run_us_rtx3050.ps1` | Drive `Cache/US` |
| M4 Air / 16 GB | France | `run_france_m4.sh` | Drive `Cache/France` |

The GPU is not required. Retrieval and feature construction mainly use CPU, RAM, and disk.

## Files to send each helper

Send each helper their shared Google Drive country folder, `test_source1.tsv`, and this repository URL:

```text
https://github.com/Mitanshp5/ML-Devs.git
```

The country folder must contain these ten files, with `US` or `France` replacing `<COUNTRY>`:

```text
dupe_map_<COUNTRY>.joblib
manifest_<COUNTRY>.json
mat_<COUNTRY>_address_only.npz
mat_<COUNTRY>_joint.npz
mat_<COUNTRY>_name_only.npz
pool_dict_<COUNTRY>.joblib
structured_index_<COUNTRY>.joblib
vec_<COUNTRY>_address_only.joblib
vec_<COUNTRY>_joint.joblib
vec_<COUNTRY>_name_only.joblib
```

Helpers do not need training data, India files, target TSVs, or another country’s cache.

## Windows RTX 3050 device — US

Open PowerShell and run:

```powershell
git clone https://github.com/Mitanshp5/ML-Devs.git
cd ML-Devs
New-Item -ItemType Directory -Force cache/retrieval_test | Out-Null
New-Item -ItemType Directory -Force student_resource/student_resource/dataset/test | Out-Null
```

Download the ten files from Drive `Cache/US` directly into:

```text
ML-Devs/cache/retrieval_test/
```

Place `test_source1.tsv` at:

```text
ML-Devs/student_resource/student_resource/dataset/test/test_source1.tsv
```

Continue in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/distributed/run_us_rtx3050.ps1
```

A healthy run prints lines similar to:

```text
Completed US_000002000: 2,000 queries, 200,000 pairs
```

Leave PowerShell open. If the process stops, run the same launcher again; completed batches are skipped automatically.

## M4 Air device — France

Open Terminal and run:

```bash
git clone https://github.com/Mitanshp5/ML-Devs.git
cd ML-Devs
mkdir -p cache/retrieval_test
mkdir -p student_resource/student_resource/dataset/test
```

Download the ten files from Drive `Cache/France`. Move the files themselves, without an additional nested `France` folder, into:

```text
ML-Devs/cache/retrieval_test/
```

Place `test_source1.tsv` at:

```text
ML-Devs/student_resource/student_resource/dataset/test/test_source1.tsv
```

Continue in Terminal:

```bash
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
bash scripts/distributed/run_france_m4.sh
```

A healthy run prints `Completed France_...` after every checkpoint. If interrupted, run the same launcher again to resume.

## Send the results back

When the launcher reports completion, return only that country’s files from `output/final/shards/`. Every completed batch has three files, and all three are required:

```text
matching_<COUNTRY>_<START>.tsv
candidates_<COUNTRY>_<START>.tsv
complete_<COUNTRY>_<START>.json
```

On the Windows US device:

```powershell
Compress-Archive -Path output/final/shards/*US* -DestinationPath US_result_shards.zip -Force
```

On the France Mac:

```bash
zip -j France_result_shards.zip output/final/shards/*France*
```

Upload the ZIP to Drive and send it to the main-device owner. Do not rename files inside it.

## Merge on the main device

Extract both returned ZIPs directly into `output/final/shards/`, alongside the locally generated India shards. Then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/distributed/merge_validate_main.ps1
```

The merge stops if any expected checkpoint is missing. After validation succeeds:

- Upload `output/final/matching_results.tsv` to the leaderboard.
- Keep `output/final/candidate_pairs.tsv` for the final competition package.

## Common errors

- `Missing required local artifact`: a cache file or `test_source1.tsv` is in the wrong directory.
- `No module named ...`: recreate the virtual environment and reinstall the requirements.
- The laptop slept or the process stopped: rerun the same country launcher to resume.
- Merge reports an incomplete shard: the country is unfinished or one of the three files for a batch was not copied back.

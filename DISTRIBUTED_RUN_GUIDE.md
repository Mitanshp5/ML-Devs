# Distributed final-inference guide

The work is split by country. Each device creates independent checkpoints, so an interrupted run can resume without losing completed batches.

| Device | Country | Launcher | Cache download |
|---|---|---|---|
| Main 32 GB Ultra 9 | India | `run_india_main.ps1` | Already local |
| Windows i5 HX / RTX 3050 / 16 GB | US | `run_us_rtx3050.ps1` | Drive `Cache/US` |
| M4 Air / 16 GB | France | `run_france_m4.sh` | Drive `Cache/France` |

The GPU is not required. Retrieval and feature construction mainly use CPU, RAM, and disk.

Use **Python 3.11** on both helper devices to match the model/cache environment and Unicode normalization. Check with `python --version` on Windows or `python3.11 --version` on macOS before creating the virtual environment.

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
.\venv\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements-inference.txt
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
python3.11 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r code/business_entity_resolution/requirements-inference.txt
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

## Optimized runner update

Before the next run, each device should pull the latest code and install the inference-specific dependencies above. All devices must use the same final bundle, code revision, original `test_source1.tsv`, and pinned numerical-library versions. Do not edit the TSV or convert its line endings.

The launch commands stay the same. India uses 12 CPU threads and RAM-backed sparse matrices; US uses 12 threads and memory-mapped sparse arrays; France uses 8 threads and RAM-backed matrices. No inference starts when you pull or install dependencies.

The US machine needs approximately **5.8 GiB additional free disk space** for prepared sparse arrays under `cache/retrieval_test/prepared_csc/`, plus room for the output shards. These prepared files stay local; do not upload them to Drive. Their first creation takes extra startup time. Memory mapping lets the OS reclaim matrix pages, but the record dictionaries and other indexes still require RAM. End-to-end memory use on the helpers has not been measured here.

Feature generation uses small blocks inside each 2,000-query checkpoint batch. Normalized records are released between blocks. Each new checkpoint records the model/input/configuration identity, exact query-range identity, output hashes, and counts. Resume verifies these before skipping work.

Older checkpoints without this identity information cannot prove which configuration created them. When their country is rerun, the runner preserves those old files in `output/final/shards/legacy/` and recomputes them. **Do not send the legacy directory back.** Send only the new top-level country shard files. If you already sent older results, rerun the updated country launcher and send the replacement files.

Do not change a country's checkpoint batch size in the same output directory. Different countries may use different batch sizes; the updated merger reads and checks their recorded ranges. A model, input, or feature-code mismatch intentionally stops resume. Use a separate output directory for a different experiment.

The merge script now uses streaming validation: every row and target ID is checked without retaining all candidate lists in RAM. It fails on missing target TSVs, duplicate rows/IDs, nonexistent targets, or matches outside the candidate set. The main device still needs all three test source TSVs for this final check; helpers only need `test_source1.tsv` for inference.

The terminal prints per-batch elapsed time and a provisional remaining-time estimate based on recent completed batches. The earlier multi-hour estimates were not full-run benchmarks. Keep the laptop plugged in and disable automatic sleep while you run your assigned launcher.

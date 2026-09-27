$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $Root
$Python = Join-Path $Root "venv/Scripts/python.exe"
if (-not (Test-Path $Python)) { throw "Missing Python environment: $Python" }
$env:PYTHONPATH = "code/business_entity_resolution/src"
& $Python -m er.run_submission_batched --bundle-dir production_bundle_final --cache-dir cache/retrieval_test --dataset-dir student_resource/student_resource/dataset --output-dir output/final --batch-size 2000 --n-cores 12 --merge-only
if ($LASTEXITCODE -ne 0) { throw "Merge failed: $LASTEXITCODE" }
& $Python -m er.validate_submission_streaming --matching output/final/matching_results.tsv --candidate output/final/candidate_pairs.tsv --test-dir student_resource/student_resource/dataset/test
if ($LASTEXITCODE -ne 0) { throw "Validation failed: $LASTEXITCODE" }
Write-Host "Validated leaderboard file: output/final/matching_results.tsv" -ForegroundColor Green

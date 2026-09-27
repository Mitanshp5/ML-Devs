$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$env:PYTHONPATH = "code/business_entity_resolution/src"
$env:OMP_NUM_THREADS = "12"
$env:MKL_NUM_THREADS = "12"
$env:OPENBLAS_NUM_THREADS = "12"
$env:NUMEXPR_NUM_THREADS = "12"
$PythonExe = Join-Path $ProjectRoot "venv/Scripts/python.exe"
$CommonArgs = @(
    "-m", "er.run_submission_batched",
    "--bundle-dir", "production_bundle_final",
    "--cache-dir", "cache/retrieval_test",
    "--dataset-dir", "student_resource/student_resource/dataset",
    "--output-dir", "output/final",
    "--batch-size", "2000",
    "--n-cores", "12",
    "--resume"
)

foreach ($Country in @("France", "India", "US")) {
    Write-Host "`n=== Starting $Country ===" -ForegroundColor Cyan
    & $PythonExe @CommonArgs --country $Country
    if ($LASTEXITCODE -ne 0) { throw "$Country inference failed with exit code $LASTEXITCODE" }
}

Write-Host "`n=== Merging completed shards ===" -ForegroundColor Cyan
& $PythonExe @CommonArgs --merge-only
if ($LASTEXITCODE -ne 0) { throw "Shard merge failed with exit code $LASTEXITCODE" }

Write-Host "`n=== Validating final submission files ===" -ForegroundColor Cyan
& $PythonExe student_resource/student_resource/utils/validate_submission.py `
    --matching output/final/matching_results.tsv `
    --candidate output/final/candidate_pairs.tsv `
    --test-dir student_resource/student_resource/dataset/test `
    --check-ids
if ($LASTEXITCODE -ne 0) { throw "Submission validation failed with exit code $LASTEXITCODE" }

Write-Host "`nFULL INFERENCE AND VALIDATION COMPLETE" -ForegroundColor Green
Write-Host "Leaderboard file: output/final/matching_results.tsv"

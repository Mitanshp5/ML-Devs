$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $Root
$Python = Join-Path $Root "venv/Scripts/python.exe"
if (-not (Test-Path $Python)) { throw "Missing venv. Create it and install code/business_entity_resolution/requirements.txt" }
foreach ($Path in @("student_resource/student_resource/dataset/test/test_source1.tsv", "cache/retrieval_test/pool_dict_India.joblib", "production_bundle_final/production_matcher_v4.txt")) {
    if (-not (Test-Path $Path)) { throw "Missing required local artifact: $Path" }
}
$env:PYTHONPATH = "code/business_entity_resolution/src"
$env:OMP_NUM_THREADS = "12"; $env:MKL_NUM_THREADS = "12"
$env:OPENBLAS_NUM_THREADS = "12"; $env:NUMEXPR_NUM_THREADS = "12"
& $Python -m er.run_submission_batched --country India --bundle-dir production_bundle_final --cache-dir cache/retrieval_test --dataset-dir student_resource/student_resource/dataset --output-dir output/final --batch-size 2000 --n-cores 12 --resume
if ($LASTEXITCODE -ne 0) { throw "India inference failed: $LASTEXITCODE" }
Write-Host "India complete. Transfer output/final/shards/*India* to the main device." -ForegroundColor Green

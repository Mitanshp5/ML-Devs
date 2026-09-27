#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if [[ -x venv/bin/python ]]; then PYTHON=venv/bin/python; elif [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
for path in student_resource/student_resource/dataset/test/test_source1.tsv cache/retrieval_test/pool_dict_France.joblib production_bundle_final/production_matcher_v4.txt; do
  [[ -f "$path" ]] || { echo "Missing required local artifact: $path" >&2; exit 1; }
done
export PYTHONPATH=code/business_entity_resolution/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
"$PYTHON" -m er.run_submission_batched --country France --bundle-dir production_bundle_final --cache-dir cache/retrieval_test --dataset-dir student_resource/student_resource/dataset --output-dir output/final --batch-size 2000 --n-cores 8 --resume
echo "France complete. Transfer output/final/shards/*France* to the main device."

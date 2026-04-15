#!/usr/bin/env bash
# Plain bash fallback for when SLURM is unavailable (local machine or HPC login node).
# Runs MLP training, SVM training, and evaluation sequentially.
#
# Usage:
#     bash scripts/run_local.sh

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

mkdir -p outputs/logs

# Ensure the dataset is present.
bash scripts/download_data.sh

echo "=== Training MLP ==="
python -m src.mlp_train --epochs 150 --batch-size 64 --lr 1e-3 --seed 42

echo "=== Training SVM (grid search) ==="
python -m src.svm_train --seed 42 --n-jobs -1

echo "=== Evaluating both models ==="
python -m src.evaluate --seed 42

echo "Done. See outputs/ for logs, figures and CSV reports."

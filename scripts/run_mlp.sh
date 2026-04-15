#!/usr/bin/env bash
# SLURM batch script: train the MLP on a GPU node.
#
# Submit from the repo root:
#     sbatch scripts/run_mlp.sh
# Monitor the queue:
#     squeue -u $USER
# Tail live log:
#     tail -f outputs/logs/mlp_*.log

#SBATCH --job-name=neco-mlp
#SBATCH --output=outputs/logs/slurm_mlp_%j.out
#SBATCH --error=outputs/logs/slurm_mlp_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

# Some clusters require loading a module before Python is available.
# Uncomment and adapt the next line if needed.
# module load python/3.10

# Activate a virtualenv created with: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

mkdir -p outputs/logs
echo "Running on $(hostname)"
nvidia-smi || true

python -m src.mlp_train --epochs 150 --batch-size 64 --lr 1e-3 --seed 42

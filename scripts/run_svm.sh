#!/usr/bin/env bash
# SLURM batch script: train the SVM on a CPU-only node with many cores.
#
# Submit from the repo root:
#     sbatch scripts/run_svm.sh
# Monitor the queue:
#     squeue -u $USER

#SBATCH --job-name=neco-svm
#SBATCH --output=outputs/logs/slurm_svm_%j.out
#SBATCH --error=outputs/logs/slurm_svm_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

# Uncomment if a module-load step is required on your cluster.
# module load python/3.10

if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

mkdir -p outputs/logs
echo "Running on $(hostname) with $SLURM_CPUS_PER_TASK CPUs"

python -m src.svm_train --seed 42 --n-jobs "$SLURM_CPUS_PER_TASK"

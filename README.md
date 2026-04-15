# NECO INM427 — Letter Recognition: MLP vs SVM

Individual coursework for City, University of London INM427 *Neural Computing*.
This repository implements and compares a PyTorch **Multilayer Perceptron (MLP)**
against a scikit-learn **Support Vector Machine (SVM)** on the UCI
**Letter Recognition** dataset (Frey & Slate, 1991).

The project is fully reproducible: every script sets random seeds, and the
required `notebooks/test.ipynb` loads the trained artifacts and re-evaluates
them on the test set **without retraining**.

---

## Repository layout

```
neco-letter-recognition/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                         # downloaded dataset (gitignored)
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # download, load, split, scale, encode
│   ├── mlp_model.py              # MLP architecture
│   ├── mlp_train.py              # training loop with early stopping
│   ├── svm_train.py              # grid search over (C, gamma)
│   ├── evaluate.py               # loads saved models, metrics, CMs, McNemar
│   └── utils.py                  # seeding, logging, plotting, McNemar
├── scripts/
│   ├── download_data.sh
│   ├── run_mlp.sh                # SLURM sbatch (1 GPU, 4 CPUs, 1 h)
│   ├── run_svm.sh                # SLURM sbatch (16 CPUs, 2 h)
│   └── run_local.sh              # plain bash fallback
├── notebooks/
│   ├── 01_exploration.ipynb      # dataset stats & feature distributions
│   ├── 02_results_analysis.ipynb # all paper figures
│   └── test.ipynb                # REQUIRED by coursework — no retraining
├── models/                       # saved artifacts (gitignored)
└── outputs/                      # figures, CSVs, logs (gitignored)
```

---

## Setup

Works on Linux (HPC/Ubuntu), macOS and Windows (WSL or native with git-bash).

```bash
# 1. Clone
git clone <your-repo-url> neco-letter-recognition
cd neco-letter-recognition

# 2. Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # (Windows cmd: venv\Scripts\activate)

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download the dataset into ./data
bash scripts/download_data.sh
```

> **GPU / CUDA note:** `requirements.txt` pins a default PyTorch wheel.
> On HPC clusters with a specific CUDA version (check `nvidia-smi` or
> `module avail cuda`) install the matching wheel from
> [pytorch.org](https://pytorch.org/get-started/locally/) **before** running
> `pip install -r requirements.txt`.

---

## How to train

### On HPC with SLURM

```bash
# from the repo root
sbatch scripts/run_mlp.sh      # 1 GPU, 4 CPUs, 8 GB, 1 h wall-time
sbatch scripts/run_svm.sh      # 16 CPUs, 8 GB, 2 h wall-time
squeue -u $USER                # monitor your jobs
tail -f outputs/logs/mlp_*.log # watch the MLP log as it runs
```

Both scripts `source venv/bin/activate` if a `venv/` exists at the repo root.
If your cluster requires a module-load step (e.g. `module load python/3.10`)
uncomment the corresponding line inside each `.sh` script.

### On HPC without SLURM / on a local machine

```bash
bash scripts/run_local.sh
```

That script runs, in order: dataset download → MLP training → SVM grid search
→ evaluation. It works identically on CPU-only machines (just slower).

You can also run the individual steps:

```bash
python -m src.mlp_train --epochs 150 --batch-size 64 --lr 1e-3 --seed 42
python -m src.svm_train --seed 42
python -m src.evaluate --seed 42
```

All train/evaluate scripts accept `--seed`, and the MLP script additionally
accepts `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--patience`,
and `--output-dir`.

---

## How to evaluate (no retraining)

Either the notebook or the script:

```bash
jupyter notebook notebooks/test.ipynb
# …or…
python -m src.evaluate
```

`notebooks/test.ipynb` is what the markers will run. It loads
`models/mlp_best.pt` + `models/svm_best.joblib` together with the saved
`scaler.joblib` and `label_encoder.joblib`, runs both models on the
Frey & Slate test split, and prints accuracy, macro-F1, per-class metrics,
confusion matrices and the McNemar statistic.

---

## Expected runtimes and accuracies

| Model | Runtime (1 GPU / 16 CPUs) | Test accuracy |
|-------|--------------------------|----------------|
| MLP   | ~2–5 min on GPU, ~10–15 min on CPU | **~93–96 %** |
| SVM   | ~5–15 min with 16 CPUs grid search | **~96–97 %** |

McNemar's test (df = 1, continuity-corrected) typically rejects equal-error
at p < 0.05.

---

## Submission metadata

```yaml
requirements:
  python: ">=3.10"
  packages:
    - torch==2.3.1
    - numpy==1.26.4
    - pandas==2.2.2
    - scikit-learn==1.5.1
    - scipy==1.13.1
    - matplotlib==3.9.1
    - seaborn==0.13.2
    - joblib==1.4.2
    - jupyter==1.0.0
    - tqdm==4.66.4
  hardware:
    gpu: "optional — CUDA speeds up MLP training, CPU fallback works"
    ram: "8 GB recommended"

setup_instructions: |
  1. git clone <this-repo> && cd neco-letter-recognition
  2. python3 -m venv venv && source venv/bin/activate
  3. pip install -r requirements.txt
  4. bash scripts/download_data.sh
  5. To reproduce training:
       - HPC+SLURM: sbatch scripts/run_mlp.sh && sbatch scripts/run_svm.sh
       - Local / no SLURM: bash scripts/run_local.sh
  6. To evaluate the saved models without retraining, open
       notebooks/test.ipynb  (or run: python -m src.evaluate)
```

---

## References

- P. W. Frey and D. J. Slate, "Letter Recognition Using Holland-style Adaptive
  Classifiers", *Machine Learning*, 6(2), 161–182, 1991.
- UCI Machine Learning Repository — Letter Recognition Data Set:
  <https://archive.ics.uci.edu/ml/datasets/letter+recognition>

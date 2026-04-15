"""Train and tune an RBF-kernel SVM on the UCI Letter Recognition dataset.

The model is a sklearn Pipeline with StandardScaler + SVC(kernel='rbf'). We run
5-fold stratified cross-validation grid search over a C x gamma grid and save
the best fitted Pipeline alongside the full cv_results_ table.

CLI
---
    python -m src.svm_train --seed 42
or
    python src/svm_train.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import prepare
    from src.utils import configure_logging, models_dir, outputs_dir, set_seed
else:
    from .data_loader import prepare
    from .utils import configure_logging, models_dir, outputs_dir, set_seed


C_GRID = [1, 5, 10, 50, 100]
GAMMA_GRID = [0.005, 0.01, 0.02, 0.05, 0.1]


def build_pipeline() -> Pipeline:
    """Construct a fresh StandardScaler + RBF-SVC pipeline.

    Note: the raw training features passed in have already been scaled by
    ``data_loader.prepare``, but we include a scaler in the saved pipeline so
    that downstream callers can pass *unscaled* features too.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=10.0, gamma=0.01, decision_function_shape="ovo")),
        ]
    )


def train(
    seed: int = 42,
    n_jobs: int = -1,
    output_dir: Optional[Path] = None,
) -> dict:
    """Grid-search SVM hyperparameters, fit best model on full training set, save artifacts."""
    set_seed(seed)
    logger = configure_logging("svm")
    logger.info("Loading dataset...")
    data = prepare()

    out_dir = Path(output_dir) if output_dir else models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # We feed the *unscaled* features to the pipeline so its internal scaler
    # becomes part of the saved artifact and downstream callers don't need to
    # remember to pre-scale inputs.
    X_train_raw = data.scaler.inverse_transform(data.X_train)
    X_test_raw = data.scaler.inverse_transform(data.X_test)

    pipeline = build_pipeline()
    param_grid = {
        "svc__C": C_GRID,
        "svc__gamma": GAMMA_GRID,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    logger.info(
        "Starting GridSearchCV: %d C values x %d gamma values, 5-fold CV, n_jobs=%s",
        len(C_GRID),
        len(GAMMA_GRID),
        n_jobs,
    )
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=n_jobs,
        verbose=2,
        refit=True,
        return_train_score=False,
    )
    t_start = time.time()
    grid.fit(X_train_raw, data.y_train)
    elapsed = time.time() - t_start
    logger.info("GridSearch finished in %.1fs. Best params: %s", elapsed, grid.best_params_)
    logger.info("Best CV accuracy: %.4f", grid.best_score_)

    best_pipeline: Pipeline = grid.best_estimator_  # type: ignore[assignment]

    # Held-out test accuracy as a sanity check.
    test_acc = float(np.mean(best_pipeline.predict(X_test_raw) == data.y_test))
    logger.info("Held-out test accuracy: %.4f", test_acc)

    joblib.dump(best_pipeline, out_dir / "svm_best.joblib")
    # Save CV results as CSV for the grid search heatmap figure.
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df.to_csv(outputs_dir() / "svm_grid_search.csv", index=False)
    # Save a compact summary for test.ipynb and the report.
    summary = {
        "best_params": grid.best_params_,
        "best_cv_accuracy": float(grid.best_score_),
        "test_accuracy": test_acc,
        "C_grid": C_GRID,
        "gamma_grid": GAMMA_GRID,
        "elapsed_seconds": elapsed,
    }
    with open(out_dir / "svm_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Saved SVM artifacts to %s", out_dir)
    return summary


def parse_args() -> argparse.Namespace:
    """CLI parser for SVM training options."""
    p = argparse.ArgumentParser(description="Train the NECO SVM model.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--output-dir", type=Path, default=None)
    # Accepted-but-ignored flags for CLI symmetry with mlp_train.py.
    p.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--lr", type=float, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(seed=args.seed, n_jobs=args.n_jobs, output_dir=args.output_dir)

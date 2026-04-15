"""Shared utilities: project paths, seeding, logging, plotting, McNemar's test.

All modules import path helpers and seeding from here so that scripts and
notebooks behave identically on any host (local Windows/macOS/Linux, HPC).
"""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Return the absolute path of the repository root.

    The root is detected by walking up from this file until we find a directory
    that contains ``requirements.txt``. This means no hardcoded absolute paths
    anywhere in the project.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    # Fallback: assume one level above src/
    return here.parent.parent


def data_dir() -> Path:
    """Directory where the raw dataset lives."""
    return project_root() / "data"


def models_dir() -> Path:
    """Directory where trained model artifacts are saved."""
    d = project_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def outputs_dir() -> Path:
    """Root directory for figures, CSVs and logs."""
    d = project_root() / "outputs"
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(parents=True, exist_ok=True)
    return d


def figures_dir() -> Path:
    """Directory where generated figures are saved."""
    return outputs_dir() / "figures"


def logs_dir() -> Path:
    """Directory where run logs are saved."""
    return outputs_dir() / "logs"


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs for reproducibility.

    Also sets the CuBLAS workspace env var to make cuDNN deterministic when
    a GPU is available.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(model_name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure a named logger that writes to both stdout and a timestamped file.

    Parameters
    ----------
    model_name: Short tag used in the log filename, e.g. ``"mlp"`` or ``"svm"``.
    log_file: Optional explicit path; if omitted a timestamped file is created
        under ``outputs/logs/``.

    Returns
    -------
    Logger ready to emit messages to both destinations.
    """
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if reconfigured inside a notebook.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir() / f"{model_name}_{ts}.log"

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Log file: %s", log_file)
    return logger


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Iterable[str],
    title: str,
    save_path: Optional[Path] = None,
    cmap: str = "Blues",
):
    """Render a 26x26 confusion matrix as a matplotlib heatmap.

    Parameters
    ----------
    cm: Square confusion-matrix array.
    class_names: Iterable of class labels used for tick labels.
    title: Figure title.
    save_path: If provided, the figure is written here (PNG).
    cmap: Matplotlib colormap name.
    """
    import matplotlib.pyplot as plt

    class_names = list(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Annotate only meaningful cells to avoid clutter on 26x26.
    threshold = cm.max() * 0.5 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                ax.text(
                    j,
                    i,
                    str(int(val)),
                    ha="center",
                    va="center",
                    color="white" if val > threshold else "black",
                    fontsize=6,
                )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_training_curves(history: dict, save_path: Optional[Path] = None):
    """Plot MLP train/validation loss and accuracy curves side by side.

    ``history`` must contain the keys ``train_loss``, ``val_loss``,
    ``train_acc`` and ``val_acc`` — one value per epoch.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["train_loss"], label="Train")
    ax_loss.plot(epochs, history["val_loss"], label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("MLP Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="Train")
    ax_acc.plot(epochs, history["val_acc"], label="Validation")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("MLP Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Compute McNemar's test (with continuity correction) comparing two classifiers.

    The test asks whether classifiers A and B disagree in a statistically
    asymmetric way on the same test instances.

    Returns
    -------
    statistic: The chi-squared statistic with continuity correction.
    p_value: The two-sided p-value under a chi-squared distribution (df=1).
    table: 2x2 contingency table:
        [[both correct, A correct & B wrong],
         [A wrong & B correct, both wrong]]
    """
    from scipy.stats import chi2

    y_true = np.asarray(y_true)
    a_correct = np.asarray(y_pred_a) == y_true
    b_correct = np.asarray(y_pred_b) == y_true

    both = int(np.sum(a_correct & b_correct))
    only_a = int(np.sum(a_correct & ~b_correct))
    only_b = int(np.sum(~a_correct & b_correct))
    neither = int(np.sum(~a_correct & ~b_correct))

    table = np.array([[both, only_a], [only_b, neither]])

    # Continuity-corrected McNemar statistic uses the off-diagonal disagreements.
    b_cell, c_cell = only_a, only_b
    denom = b_cell + c_cell
    if denom == 0:
        return 0.0, 1.0, table
    stat = (abs(b_cell - c_cell) - 1) ** 2 / denom
    p_value = float(1 - chi2.cdf(stat, df=1))
    return float(stat), p_value, table


LETTER_CLASSES: Tuple[str, ...] = tuple(chr(ord("A") + i) for i in range(26))
"""Tuple of class names 'A'..'Z' used throughout the project."""

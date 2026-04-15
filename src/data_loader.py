"""UCI Letter Recognition dataset loader.

Responsibilities:
- Download the raw CSV if not already present.
- Parse into features/labels with deterministic column names.
- Apply the Frey & Slate (1991) split: first 16,000 rows train, last 4,000 test.
- Label-encode letters A-Z to integers 0-25.
- Fit a StandardScaler on training data only.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .utils import data_dir

# UCI Letter Recognition: 16 integer features + 1 letter label.
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "letter-recognition/letter-recognition.data"
)
FEATURE_COLUMNS = [
    "x-box",
    "y-box",
    "width",
    "high",
    "onpix",
    "x-bar",
    "y-bar",
    "x2bar",
    "y2bar",
    "xybar",
    "x2ybr",
    "xy2br",
    "x-ege",
    "xegvy",
    "y-ege",
    "yedgex",
]
LABEL_COLUMN = "letter"
TRAIN_SIZE = 16000
TOTAL_SIZE = 20000

logger = logging.getLogger(__name__)


@dataclass
class LetterData:
    """Container for a fully preprocessed dataset split.

    Attributes
    ----------
    X_train, X_test: scaled feature matrices (float32).
    y_train, y_test: integer-encoded labels 0..25.
    scaler: the fitted StandardScaler (fit on X_train only).
    encoder: the fitted LabelEncoder.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    encoder: LabelEncoder


def download_dataset(target: Optional[Path] = None) -> Path:
    """Download the UCI Letter Recognition CSV if it does not exist.

    Parameters
    ----------
    target: Optional explicit destination path. Defaults to
        ``<project>/data/letter-recognition.data``.

    Returns
    -------
    Path to the downloaded file.
    """
    if target is None:
        target = data_dir() / "letter-recognition.data"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        logger.info("Dataset already present at %s", target)
        return target
    logger.info("Downloading UCI Letter Recognition from %s", DATA_URL)
    urllib.request.urlretrieve(DATA_URL, target)
    logger.info("Saved dataset to %s", target)
    return target


def load_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame with named columns.

    The CSV is comma-separated, with the letter label in column 0 followed by
    16 integer feature columns.
    """
    if path is None:
        path = data_dir() / "letter-recognition.data"
    if not path.exists():
        path = download_dataset(path)
    df = pd.read_csv(path, header=None, names=[LABEL_COLUMN, *FEATURE_COLUMNS])
    if len(df) != TOTAL_SIZE:
        logger.warning("Expected %d rows, got %d", TOTAL_SIZE, len(df))
    return df


def prepare(
    path: Optional[Path] = None,
    train_size: int = TRAIN_SIZE,
) -> LetterData:
    """Load, split, encode and scale the dataset.

    The split matches Frey & Slate (1991): the first ``train_size`` rows form
    the training set and the remainder are the held-out test set. The
    ``StandardScaler`` is fit on the training features only, then applied to
    both splits.
    """
    df = load_raw(path)

    # Integer-encode A..Z once; ``fit`` runs on the full label column so the
    # encoder covers every class regardless of ordering.
    encoder = LabelEncoder()
    encoder.fit(df[LABEL_COLUMN])

    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    X_train_raw = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    X_test_raw = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = encoder.transform(train_df[LABEL_COLUMN]).astype(np.int64)
    y_test = encoder.transform(test_df[LABEL_COLUMN]).astype(np.int64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    logger.info(
        "Prepared dataset: train=%d, test=%d, n_features=%d, n_classes=%d",
        len(X_train),
        len(X_test),
        X_train.shape[1],
        len(encoder.classes_),
    )
    return LetterData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        encoder=encoder,
    )


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic stratified-by-index train/validation split.

    A simple shuffled split is sufficient here because the dataset is
    balanced across 26 classes.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    n_val = int(len(X) * val_fraction)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

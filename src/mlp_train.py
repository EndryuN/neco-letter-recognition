"""Train the MLP on the UCI Letter Recognition dataset.

Implements:
- Adam optimizer (lr=1e-3, weight_decay=1e-4), CrossEntropyLoss.
- Batch size 64, up to 150 epochs, early stopping on validation loss (patience=15).
- 10% of the training set reserved for validation.
- Device-agnostic: auto-detects CUDA, falls back to CPU.
- Saves the state_dict, fitted scaler+encoder, and training history to ``models/``.

CLI
---
    python -m src.mlp_train --epochs 150 --batch-size 64 --lr 1e-3 --seed 42
or
    python src/mlp_train.py --epochs 150 --batch-size 64 --lr 1e-3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Support both "python -m src.mlp_train" and "python src/mlp_train.py"
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import prepare, train_val_split
    from src.mlp_model import MLPClassifier
    from src.utils import (
        LETTER_CLASSES,
        configure_logging,
        models_dir,
        set_seed,
    )
else:
    from .data_loader import prepare, train_val_split
    from .mlp_model import MLPClassifier
    from .utils import LETTER_CLASSES, configure_logging, models_dir, set_seed


def select_device() -> torch.device:
    """Return the best available torch device (CUDA if present, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Wrap train/val arrays in PyTorch DataLoaders."""
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, float]:
    """Run a single epoch in either train or eval mode.

    Returns the mean loss and accuracy across the loader.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_count += len(yb)
    return total_loss / total_count, total_correct / total_count


def train(
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> dict:
    """Full MLP training pipeline with early stopping.

    Returns the history dictionary (also saved alongside the model).
    """
    set_seed(seed)
    logger = configure_logging("mlp")
    device = select_device()
    logger.info("Using device: %s", device)

    out_dir = Path(output_dir) if output_dir else models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = prepare()
    X_tr, X_val, y_tr, y_val = train_val_split(
        data.X_train, data.y_train, val_fraction=0.1, seed=seed
    )
    train_loader, val_loader = build_loaders(X_tr, y_tr, X_val, y_val, batch_size)

    model = MLPClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0

    logger.info("Starting training: epochs=%d batch_size=%d lr=%.4g", epochs, batch_size, lr)
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        logger.info(
            "Epoch %3d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f",
            epoch,
            tr_loss,
            tr_acc,
            val_loss,
            val_acc,
        )
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs).", epoch, patience)
                break
    elapsed = time.time() - t_start
    logger.info("Training finished in %.1fs. Best epoch=%d val_loss=%.4f", elapsed, best_epoch, best_val_loss)

    assert best_state is not None, "Training produced no best checkpoint"
    model.load_state_dict(best_state)

    # Persist everything test.ipynb needs to reload the model without retraining.
    state_path = out_dir / "mlp_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": 16,
            "hidden_dims": (128, 64),
            "num_classes": 26,
            "dropout": 0.2,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        state_path,
    )
    joblib.dump(data.scaler, out_dir / "scaler.joblib")
    joblib.dump(data.encoder, out_dir / "label_encoder.joblib")
    with open(out_dir / "mlp_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    logger.info("Saved MLP artifacts to %s", out_dir)

    # Final test-set evaluation for quick sanity check.
    model.eval()
    with torch.no_grad():
        X_te = torch.from_numpy(data.X_test).to(device)
        y_te = torch.from_numpy(data.y_test).to(device)
        logits = model(X_te)
        test_acc = float((logits.argmax(dim=1) == y_te).float().mean().item())
    logger.info("Held-out test accuracy: %.4f (n=%d classes=%d)", test_acc, len(data.y_test), len(LETTER_CLASSES))

    history["best_epoch"] = best_epoch
    history["test_accuracy"] = test_acc
    return history


def parse_args() -> argparse.Namespace:
    """CLI parser for training hyperparameters."""
    p = argparse.ArgumentParser(description="Train the NECO MLP model.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        output_dir=args.output_dir,
    )

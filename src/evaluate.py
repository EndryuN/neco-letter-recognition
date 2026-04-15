"""Load saved MLP + SVM artifacts and evaluate them on the held-out test set.

Produces:
- Overall accuracy, macro F1, per-class precision/recall/F1 (CSV).
- 26x26 confusion matrices (PNG) for both models.
- McNemar's test comparing MLP vs SVM predictions.

Can be called as a module or a script:

    python -m src.evaluate
    python src/evaluate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import prepare
    from src.mlp_model import MLPClassifier
    from src.utils import (
        LETTER_CLASSES,
        configure_logging,
        figures_dir,
        mcnemar_test,
        models_dir,
        outputs_dir,
        plot_confusion_matrix,
        set_seed,
    )
else:
    from .data_loader import prepare
    from .mlp_model import MLPClassifier
    from .utils import (
        LETTER_CLASSES,
        configure_logging,
        figures_dir,
        mcnemar_test,
        models_dir,
        outputs_dir,
        plot_confusion_matrix,
        set_seed,
    )


def load_mlp(models_root: Optional[Path] = None) -> tuple[MLPClassifier, torch.device]:
    """Instantiate the MLP and load its saved state dict."""
    models_root = Path(models_root) if models_root else models_dir()
    ckpt = torch.load(models_root / "mlp_best.pt", map_location="cpu", weights_only=False)
    model = MLPClassifier(
        input_dim=ckpt.get("input_dim", 16),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        num_classes=ckpt.get("num_classes", 26),
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def mlp_predict(model: MLPClassifier, device: torch.device, X: np.ndarray) -> np.ndarray:
    """Return class predictions from the MLP for a batch of (already scaled) inputs."""
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        logits = model(xb)
        return logits.argmax(dim=1).cpu().numpy()


def evaluate(models_root: Optional[Path] = None, seed: int = 42) -> dict:
    """Run the full evaluation pipeline and return a summary dictionary."""
    set_seed(seed)
    logger = configure_logging("evaluate")
    models_root = Path(models_root) if models_root else models_dir()
    data = prepare()

    # --- MLP --------------------------------------------------------------
    logger.info("Loading MLP from %s", models_root / "mlp_best.pt")
    mlp_model, device = load_mlp(models_root)
    mlp_pred = mlp_predict(mlp_model, device, data.X_test)
    mlp_acc = accuracy_score(data.y_test, mlp_pred)
    mlp_f1 = f1_score(data.y_test, mlp_pred, average="macro")
    logger.info("MLP test accuracy=%.4f macro_f1=%.4f", mlp_acc, mlp_f1)

    # --- SVM --------------------------------------------------------------
    logger.info("Loading SVM from %s", models_root / "svm_best.joblib")
    svm = joblib.load(models_root / "svm_best.joblib")
    # The saved SVM pipeline includes its own scaler, so we pass raw features.
    X_test_raw = data.scaler.inverse_transform(data.X_test)
    svm_pred = svm.predict(X_test_raw)
    svm_acc = accuracy_score(data.y_test, svm_pred)
    svm_f1 = f1_score(data.y_test, svm_pred, average="macro")
    logger.info("SVM test accuracy=%.4f macro_f1=%.4f", svm_acc, svm_f1)

    # --- Classification reports -----------------------------------------
    class_names = list(LETTER_CLASSES)
    out_dir = outputs_dir()
    mlp_report = classification_report(
        data.y_test, mlp_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    svm_report = classification_report(
        data.y_test, svm_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(mlp_report).T.to_csv(out_dir / "mlp_classification_report.csv")
    pd.DataFrame(svm_report).T.to_csv(out_dir / "svm_classification_report.csv")

    # --- Confusion matrices --------------------------------------------
    cm_mlp = confusion_matrix(data.y_test, mlp_pred, labels=range(26))
    cm_svm = confusion_matrix(data.y_test, svm_pred, labels=range(26))
    plot_confusion_matrix(
        cm_mlp,
        class_names,
        "MLP Confusion Matrix",
        save_path=figures_dir() / "confusion_matrix_mlp.png",
    )
    plot_confusion_matrix(
        cm_svm,
        class_names,
        "SVM Confusion Matrix",
        save_path=figures_dir() / "confusion_matrix_svm.png",
    )

    # --- McNemar -------------------------------------------------------
    stat, p_value, table = mcnemar_test(data.y_test, mlp_pred, svm_pred)
    logger.info("McNemar chi2=%.4f p=%.4g", stat, p_value)
    logger.info("Contingency table (both, only-MLP, only-SVM, neither) = %s", table.tolist())

    summary = {
        "mlp": {"accuracy": float(mlp_acc), "macro_f1": float(mlp_f1)},
        "svm": {"accuracy": float(svm_acc), "macro_f1": float(svm_f1)},
        "mcnemar": {
            "statistic": float(stat),
            "p_value": float(p_value),
            "table": table.tolist(),
        },
    }
    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote summary to %s", out_dir / "evaluation_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    """CLI parser for evaluation options."""
    p = argparse.ArgumentParser(description="Evaluate saved MLP and SVM models.")
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(models_root=args.models_dir, seed=args.seed)

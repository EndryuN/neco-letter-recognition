"""PyTorch MLP architecture for the UCI Letter Recognition task.

Two hidden layers (128 -> 64) with ReLU and dropout 0.2, plus a 26-unit output
layer returning raw logits (CrossEntropyLoss expects logits).

He/Kaiming initialization is applied to linear weights; biases initialize to 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Multilayer perceptron for 26-class letter classification.

    Parameters
    ----------
    input_dim: Number of input features (default 16).
    hidden_dims: Tuple with the two hidden layer widths.
    num_classes: Number of output classes (default 26).
    dropout: Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: tuple[int, int] = (128, 64),
        num_classes: int = 26,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(h2, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming (He) initialization to linear weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for each class."""
        return self.net(x)

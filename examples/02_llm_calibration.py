"""
Example 2: LLM output calibration with WassersteinLoss
======================================================
Replaces cross-entropy with Wasserstein distance for classification,
producing probability outputs that are better calibrated (confidence
matches accuracy).

Key insight: WassersteinLoss treats class labels as points on a metric
space, so predicting class 3 when truth is class 4 incurs less loss than
predicting class 0. Cross-entropy treats all errors equally.

Run: python examples/02_llm_calibration.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from otloss import WassersteinLoss, calibration_error
from otloss.distributions import label_smoothed_weights


# ---------------------------------------------------------------------------
# Toy classifier on synthetic data
# ---------------------------------------------------------------------------

def make_dataset(n: int = 2000, n_classes: int = 10):
    torch.manual_seed(0)
    X = torch.randn(n, 32)
    y = torch.randint(0, n_classes, (n,))
    # Add class-correlated signal
    for c in range(n_classes):
        mask = y == c
        X[mask, c * 3 % 32:(c * 3 % 32) + 3] += 2.0
    return X, y


class Classifier(nn.Module):
    def __init__(self, in_dim: int = 32, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
            nn.Linear(64, n_classes),
        )
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training with cross-entropy (baseline)
# ---------------------------------------------------------------------------

def train_cross_entropy(X, y, n_classes: int = 10, n_epochs: int = 100):
    model = Classifier(n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(n_epochs):
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# ---------------------------------------------------------------------------
# Training with WassersteinLoss
# ---------------------------------------------------------------------------

def train_wasserstein(X, y, n_classes: int = 10, n_epochs: int = 100):
    """
    The key difference: we treat class probabilities as a 1-D distribution
    over the class index space. Wasserstein distance measures how far the
    predicted distribution is from the label distribution — using the
    ordinal structure of class indices as the ground metric.

    Setup:
        pred  → softmax(logits) treated as weights on {0,1,...,K-1}
        target → label-smoothed one-hot on same support
        support → class index embeddings (K, 1)
    """
    model = Classifier(n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = WassersteinLoss(p=2, blur=0.1, debias=True)

    # Class support: ordinal positions normalised to [0, 1]
    class_positions = torch.linspace(0, 1, n_classes).unsqueeze(-1)  # (K, 1)
    # Expand to batch: (B, K, 1)

    for epoch in range(n_epochs):
        logits = model(X)
        pred_probs = torch.softmax(logits, dim=-1)   # (B, K)

        # Target: label-smoothed one-hot → (B, K)
        target_weights = label_smoothed_weights(y, n_classes, smoothing=0.1)

        # Expand class positions to batch
        B = X.shape[0]
        support = class_positions.unsqueeze(0).expand(B, -1, -1)  # (B, K, 1)

        # WassersteinLoss: measure W₂ between pred and target on class support
        loss = criterion(
            pred=support,
            target=support,
            pred_weights=pred_probs,
            target_weights=target_weights,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | W₂ loss: {loss.item():.6f}")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, name: str):
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=-1)
        acc = (probs.argmax(-1) == y).float().mean()
        ece = calibration_error(probs, y)
    print(f"{name:20s} | Accuracy: {acc:.3f} | ECE: {ece:.4f}")


if __name__ == "__main__":
    X, y = make_dataset(n=2000, n_classes=10)
    X_train, y_train = X[:1600], y[:1600]
    X_test, y_test   = X[1600:], y[1600:]

    print("Training cross-entropy baseline...")
    ce_model = train_cross_entropy(X_train, y_train, n_epochs=200)

    print("\nTraining WassersteinLoss model...")
    w_model = train_wasserstein(X_train, y_train, n_epochs=200)

    print("\n--- Test set results ---")
    evaluate(ce_model, X_test, y_test, "Cross-entropy")
    evaluate(w_model,  X_test, y_test, "WassersteinLoss")
    print("\nExpected: WassersteinLoss achieves lower ECE (better calibration)")

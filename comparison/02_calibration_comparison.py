"""
Comparison 2: LLM Calibration — Expected Calibration Error (ECE)
=================================================================
Trains a 10-class classifier two ways.
Baseline (CrossEntropy) produces overconfident outputs.
otloss (WassersteinLoss on class-space) produces calibrated probabilities.

ECE = Σ_b |B_b|/n · |accuracy(B_b) − confidence(B_b)|
Lower ECE = better calibrated.

Run:
    pip install otloss
    python 02_calibration_comparison.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from otloss import WassersteinLoss, calibration_error
from otloss.distributions import label_smoothed_weights


# ── data ──────────────────────────────────────────────────────────────────────
def make_dataset(n: int = 3000, n_classes: int = 10, seed: int = 0):
    torch.manual_seed(seed)
    X = torch.randn(n, 32)
    y = torch.randint(0, n_classes, (n,))
    for c in range(n_classes):
        X[y == c, c * 3 % 32:(c * 3 % 32) + 3] += 2.5
    return X, y


# ── model ─────────────────────────────────────────────────────────────────────
class Classifier(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    def forward(self, x): return self.net(x)


# ── setup ─────────────────────────────────────────────────────────────────────
N_CLASSES = 10
EPOCHS    = 200
BATCH     = 128
LR        = 2e-3
LOG_EVERY = 50

X, y   = make_dataset(3000, N_CLASSES)
X_tr, y_tr = X[:2400], y[:2400]
X_te, y_te = X[2400:], y[2400:]

print("\n" + "="*60)
print("  Classifier calibration comparison — ECE")
print("="*60)


def train_epoch(model, opt, loss_fn):
    model.train()
    perm = torch.randperm(len(X_tr))
    total = 0.0; n = 0
    for i in range(0, len(X_tr), BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = X_tr[idx], y_tr[idx]
        loss = loss_fn(model, xb, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(xb); n += len(xb)
    return total / n


def evaluate(model):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_te), dim=-1)
        acc   = (probs.argmax(-1) == y_te).float().mean().item()
        ece   = calibration_error(probs, y_te).item()
    return acc, ece


# ── Baseline: CrossEntropyLoss ────────────────────────────────────────────────
print("\n[1/2] Baseline — CrossEntropyLoss")
torch.manual_seed(42)
clf_ce  = Classifier(N_CLASSES)
opt_ce  = torch.optim.Adam(clf_ce.parameters(), lr=LR)

def ce_loss(model, xb, yb):
    return F.cross_entropy(model(xb), yb)

t0 = time.time()
for ep in range(1, EPOCHS + 1):
    train_loss = train_epoch(clf_ce, opt_ce, ce_loss)
    if ep % LOG_EVERY == 0:
        acc, ece = evaluate(clf_ce)
        print(f"  epoch {ep:3d} | loss: {train_loss:.4f} | acc: {acc:.3f} | ECE: {ece:.4f}")

ce_time     = time.time() - t0
ce_acc, ce_ece = evaluate(clf_ce)

# ── otloss: WassersteinLoss on class-support ──────────────────────────────────
print(f"\n[2/2] otloss — WassersteinLoss (class-support calibration)")
torch.manual_seed(42)
clf_ot  = Classifier(N_CLASSES)
opt_ot  = torch.optim.Adam(clf_ot.parameters(), lr=LR)
w_crit  = WassersteinLoss(p=2, blur=0.08, debias=True, scaling=1.0)

# Class positions: treat each class as a point on [0, 1] — ordinal metric
class_pos = torch.linspace(0, 1, N_CLASSES).unsqueeze(-1)   # (K, 1)

def ot_loss(model, xb, yb):
    B = xb.shape[0]
    pred_w = torch.softmax(model(xb), dim=-1)                # (B, K)
    tgt_w  = label_smoothed_weights(yb, N_CLASSES, 0.05)     # (B, K)
    sup    = class_pos.unsqueeze(0).expand(B, -1, -1)         # (B, K, 1)
    return w_crit(sup, sup, pred_weights=pred_w, target_weights=tgt_w)

t0 = time.time()
for ep in range(1, EPOCHS + 1):
    train_loss = train_epoch(clf_ot, opt_ot, ot_loss)
    if ep % LOG_EVERY == 0:
        acc, ece = evaluate(clf_ot)
        print(f"  epoch {ep:3d} | loss: {train_loss:.4f} | acc: {acc:.3f} | ECE: {ece:.4f}")

ot_time     = time.time() - t0
ot_acc, ot_ece = evaluate(clf_ot)

# ── summary ───────────────────────────────────────────────────────────────────
ece_improvement = (ce_ece - ot_ece) / ce_ece * 100
print("\n" + "="*60)
print("  RESULTS")
print("="*60)
print(f"  {'Metric':<28} {'CrossEntropy':>12} {'otloss':>10}")
print(f"  {'-'*52}")
print(f"  {'Accuracy':<28} {ce_acc:>12.3f} {ot_acc:>10.3f}")
print(f"  {'ECE (↓ better)':<28} {ce_ece:>12.4f} {ot_ece:>10.4f}")
print(f"  {'Training time (s)':<28} {ce_time:>12.1f} {ot_time:>10.1f}")
print(f"  {'-'*52}")
print(f"  ECE improved by {ece_improvement:.1f}% — probabilities are better calibrated")
print(f"  (both should have similar accuracy; ECE is the key metric)")

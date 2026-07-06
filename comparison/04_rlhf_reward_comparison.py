"""
Comparison 4: RLHF Reward Model — Preference Ranking Quality
=============================================================
Simulates reward model training from human preference data.
Baseline: pointwise MSE on scalar reward scores.
otloss:   WassersteinLoss on the reward distribution — smoother landscape.

Metric: Kendall's τ (rank correlation between predicted and true rewards).
Higher τ = better preference ranking = better RLHF signal.

Run:
    pip install otloss
    python 04_rlhf_reward_comparison.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from otloss import WassersteinLoss


# ── synthetic preference data ─────────────────────────────────────────────────
def make_preference_data(n: int = 2000, response_dim: int = 16, seed: int = 0):
    """
    Simulate responses and their ground-truth reward scores.
    In real RLHF: responses are LLM outputs, rewards come from human raters.
    """
    torch.manual_seed(seed)
    responses     = torch.randn(n, response_dim)
    # True reward: sparse linear function of response features
    true_weights  = torch.zeros(response_dim)
    true_weights[:4] = torch.tensor([2.0, -1.5, 1.0, -0.5])
    true_rewards  = responses @ true_weights + torch.randn(n) * 0.3
    return responses, true_rewards


def kendall_tau(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Kendall's τ rank correlation — the key RLHF metric."""
    n = len(pred)
    pred_np = pred.detach().numpy()
    true_np = true.numpy()
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            p_diff = pred_np[i] - pred_np[j]
            t_diff = true_np[i] - true_np[j]
            if p_diff * t_diff > 0:
                concordant += 1
            elif p_diff * t_diff < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


# ── model ─────────────────────────────────────────────────────────────────────
class RewardModel(nn.Module):
    def __init__(self, input_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),        nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


# ── setup ─────────────────────────────────────────────────────────────────────
N_TRAIN   = 1600
N_TEST    = 400
DIM       = 16
EPOCHS    = 200
BATCH     = 64
LR        = 1e-3
LOG_EVERY = 40

responses, true_rewards = make_preference_data(N_TRAIN + N_TEST, DIM)
X_tr, y_tr = responses[:N_TRAIN],  true_rewards[:N_TRAIN]
X_te, y_te = responses[N_TRAIN:],  true_rewards[N_TRAIN:]

# Normalise rewards to [0, 1] for otloss support
y_min, y_max = y_tr.min(), y_tr.max()
y_tr_norm = (y_tr - y_min) / (y_max - y_min)
y_te_norm = (y_te - y_min) / (y_max - y_min)

print("\n" + "="*60)
print("  RLHF reward model — Kendall τ ranking quality")
print("="*60)


def evaluate(model, subset_size: int = 200) -> tuple[float, float]:
    """Compute MSE and Kendall τ on test set."""
    model.eval()
    idx = torch.randperm(N_TEST)[:subset_size]
    with torch.no_grad():
        pred = model(X_te[idx])
    mse = F.mse_loss(pred, y_te[idx]).item()
    tau = kendall_tau(pred, y_te[idx])
    return mse, tau


# ── Baseline: MSELoss on scalar reward ───────────────────────────────────────
print("\n[1/2] Baseline — MSELoss (pointwise reward regression)")
torch.manual_seed(42)
RM_base = RewardModel(DIM)
opt     = torch.optim.Adam(RM_base.parameters(), lr=LR)

t0 = time.time()
for ep in range(1, EPOCHS + 1):
    RM_base.train()
    perm = torch.randperm(N_TRAIN)
    ep_loss = 0.0
    for i in range(0, N_TRAIN, BATCH):
        idx  = perm[i:i + BATCH]
        pred = RM_base(X_tr[idx])
        loss = F.mse_loss(pred, y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss += loss.item()
    if ep % LOG_EVERY == 0:
        mse, tau = evaluate(RM_base)
        print(f"  epoch {ep:3d} | MSE: {mse:.4f} | Kendall τ: {tau:.4f}")

base_time = time.time() - t0
base_mse, base_tau = evaluate(RM_base, N_TEST)

# ── otloss: WassersteinLoss on reward distribution ───────────────────────────
print(f"\n[2/2] otloss — WassersteinLoss on reward distribution")
torch.manual_seed(42)
RM_ot = RewardModel(DIM)
opt   = torch.optim.Adam(RM_ot.parameters(), lr=LR)
crit  = WassersteinLoss(p=2, blur=0.05, debias=True, scaling=1.0)

t0 = time.time()
for ep in range(1, EPOCHS + 1):
    RM_ot.train()
    perm = torch.randperm(N_TRAIN)
    ep_loss = 0.0
    for i in range(0, N_TRAIN, BATCH):
        idx  = perm[i:i + BATCH]
        B    = len(idx)
        # Predicted reward as 1-D distribution (score as position on [0,1])
        pred_scores = torch.sigmoid(RM_ot(X_tr[idx]))          # (B,) in [0,1]
        tgt_scores  = y_tr_norm[idx]                            # (B,) in [0,1]
        # Treat each scalar as a point on R^1
        pred_pts = pred_scores.unsqueeze(-1).unsqueeze(0)       # (1, B, 1)
        tgt_pts  = tgt_scores.unsqueeze(-1).unsqueeze(0)        # (1, B, 1)
        loss = crit(pred_pts, tgt_pts)
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss += loss.item()
    if ep % LOG_EVERY == 0:
        mse, tau = evaluate(RM_ot)
        print(f"  epoch {ep:3d} | W2-loss: {ep_loss:.4f} | Kendall τ: {tau:.4f}")

ot_time = time.time() - t0
ot_mse, ot_tau = evaluate(RM_ot, N_TEST)

# ── summary ───────────────────────────────────────────────────────────────────
tau_improvement = (ot_tau - base_tau) / abs(base_tau) * 100 if base_tau != 0 else float('inf')
print("\n" + "="*60)
print("  RESULTS")
print("="*60)
print(f"  {'Metric':<32} {'MSELoss':>10} {'otloss':>10}")
print(f"  {'-'*54}")
print(f"  {'Test MSE (↓ better)':<32} {base_mse:>10.4f} {ot_mse:>10.4f}")
print(f"  {'Kendall τ (↑ better)':<32} {base_tau:>10.4f} {ot_tau:>10.4f}")
print(f"  {'Training time (s)':<32} {base_time:>10.1f} {ot_time:>10.1f}")
print(f"  {'-'*54}")
print(f"  Kendall τ improved by {tau_improvement:.1f}%")
print(f"  Better τ = better preference ranking = stronger RLHF signal")

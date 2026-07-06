"""
Comparison 3: Distribution Matching — Convergence Speed
========================================================
Trains a small network to map Gaussian noise → 5-cluster mixture.
Measures how fast and how accurately each loss converges.

Baseline: MSELoss — fights the geometry, slow convergence
otloss:   SlicedWassersteinLoss — geometry-aware, faster & sharper

Run:
    pip install otloss
    python 03_convergence_comparison.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from otloss import SlicedWassersteinLoss


# ── target distribution: 5-cluster mixture ────────────────────────────────────
CENTERS = torch.tensor([
    [ 2.0,  2.0],
    [-2.0,  2.0],
    [ 0.0, -2.5],
    [ 2.0, -2.0],
    [-2.0, -2.0],
])

def target_dist(n: int) -> torch.Tensor:
    idx = torch.randint(0, 5, (n,))
    return CENTERS[idx] + torch.randn(n, 2) * 0.25


# ── model ─────────────────────────────────────────────────────────────────────
class Mapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 2),
        )
    def forward(self, z): return self.net(z)


# ── metrics ───────────────────────────────────────────────────────────────────
def nearest_neighbour_dist(pred: torch.Tensor, real: torch.Tensor) -> float:
    """Mean distance from each generated point to its nearest real point."""
    with torch.no_grad():
        d = torch.cdist(pred.unsqueeze(0), real.unsqueeze(0)).squeeze(0)
    return d.min(dim=1).values.mean().item()

def cluster_coverage(M: nn.Module, n: int = 3000, threshold: float = 0.6) -> int:
    """Count how many of the 5 target clusters have ≥3% of generated samples."""
    with torch.no_grad():
        pred = M(torch.randn(n, 4)).numpy()
    covered = 0
    for cx, cy in CENTERS.numpy():
        dists = ((pred[:, 0] - cx) ** 2 + (pred[:, 1] - cy) ** 2) ** 0.5
        if (dists < threshold).sum() > n * 0.03:
            covered += 1
    return covered


# ── training ──────────────────────────────────────────────────────────────────
STEPS     = 500
BATCH     = 256
LR        = 3e-3
LOG_EVERY = 50

print("\n" + "="*60)
print("  Distribution matching — convergence speed")
print("="*60)

# ── Baseline: MSELoss ─────────────────────────────────────────────────────────
print("\n[1/2] Baseline — MSELoss")
torch.manual_seed(42)
M_base = Mapper()
opt    = torch.optim.Adam(M_base.parameters(), lr=LR)

base_log = []
t0 = time.time()
for step in range(1, STEPS + 1):
    tgt  = target_dist(BATCH)
    pred = M_base(torch.randn(BATCH, 4))
    # Shuffle target — MSE has no notion of distributional structure
    loss = F.mse_loss(pred, tgt[torch.randperm(BATCH)])
    opt.zero_grad(); loss.backward(); opt.step()

    if step % LOG_EVERY == 0:
        with torch.no_grad():
            p = M_base(torch.randn(1000, 4))
            t = target_dist(1000)
        nd = nearest_neighbour_dist(p, t)
        cc = cluster_coverage(M_base)
        base_log.append((step, nd, cc))
        print(f"  step {step:4d} | nn-dist: {nd:.4f} | clusters: {cc}/5 | loss: {loss.item():.4f}")

base_time = time.time() - t0
with torch.no_grad():
    base_final_nd = nearest_neighbour_dist(M_base(torch.randn(2000, 4)), target_dist(2000))
base_final_cc = cluster_coverage(M_base)

# ── otloss: SlicedWassersteinLoss ─────────────────────────────────────────────
print(f"\n[2/2] otloss — SlicedWassersteinLoss (n_projections=200)")
torch.manual_seed(42)
M_ot = Mapper()
opt  = torch.optim.Adam(M_ot.parameters(), lr=LR)
swd  = SlicedWassersteinLoss(n_projections=200, p=2)

ot_log = []
t0 = time.time()
for step in range(1, STEPS + 1):
    tgt  = target_dist(BATCH)
    pred = M_ot(torch.randn(BATCH, 4))
    loss = swd(pred.unsqueeze(0), tgt.unsqueeze(0))
    opt.zero_grad(); loss.backward(); opt.step()

    if step % LOG_EVERY == 0:
        with torch.no_grad():
            p = M_ot(torch.randn(1000, 4))
            t = target_dist(1000)
        nd = nearest_neighbour_dist(p, t)
        cc = cluster_coverage(M_ot)
        ot_log.append((step, nd, cc))
        print(f"  step {step:4d} | nn-dist: {nd:.4f} | clusters: {cc}/5 | loss: {loss.item():.4f}")

ot_time = time.time() - t0
with torch.no_grad():
    ot_final_nd = nearest_neighbour_dist(M_ot(torch.randn(2000, 4)), target_dist(2000))
ot_final_cc = cluster_coverage(M_ot)

# ── convergence speed: steps to reach nn-dist < 0.3 ──────────────────────────
THRESHOLD = 0.3
base_steps_to_thresh = next((s for s, nd, _ in base_log if nd < THRESHOLD), None)
ot_steps_to_thresh   = next((s for s, nd, _ in ot_log   if nd < THRESHOLD), None)

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  RESULTS")
print("="*60)
print(f"  {'Metric':<32} {'MSELoss':>10} {'otloss':>10}")
print(f"  {'-'*54}")
print(f"  {'Final nn-dist (↓ better)':<32} {base_final_nd:>10.4f} {ot_final_nd:>10.4f}")
print(f"  {'Clusters covered (max 5)':<32} {base_final_cc:>10} {ot_final_cc:>10}")

if base_steps_to_thresh:
    print(f"  {'Steps to nn-dist < 0.3':<32} {base_steps_to_thresh:>10} {ot_steps_to_thresh or 'N/A':>10}")
else:
    print(f"  {'Steps to nn-dist < 0.3':<32} {'never':>10} {ot_steps_to_thresh or 'N/A':>10}")

print(f"  {'Training time (s)':<32} {base_time:>10.1f} {ot_time:>10.1f}")
print(f"  {'-'*54}")
nd_gain = (base_final_nd - ot_final_nd) / base_final_nd * 100
print(f"  otloss reduced nn-dist by {nd_gain:.1f}%")
print(f"  otloss covered {ot_final_cc - base_final_cc} more clusters")

print("\n  Per-step convergence trace:")
print(f"  {'Step':>6}  {'MSE nn-dist':>12}  {'SWD nn-dist':>12}  {'Advantage':>10}")
print(f"  {'-'*46}")
for (bs, bnd, _), (os, ond, _) in zip(base_log, ot_log):
    adv = bnd - ond
    marker = " ◀ otloss better" if adv > 0.01 else ""
    print(f"  {bs:>6}  {bnd:>12.4f}  {ond:>12.4f}  {adv:>+10.4f}{marker}")

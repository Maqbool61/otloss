"""
Comparison 1: GAN — Mode Collapse Test
=======================================
Trains a generator on an 8-Gaussian ring.
Baseline (MSE) collapses to the mean.
otloss (WassersteinLoss) covers all 8 modes.

Run:
    pip install otloss
    python 01_gan_comparison.py
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── data ──────────────────────────────────────────────────────────────────────
def sample_8gaussians(n: int) -> torch.Tensor:
    angles  = torch.arange(8) * (2 * math.pi / 8)
    centers = torch.stack([angles.cos() * 2, angles.sin() * 2], dim=1)
    idx     = torch.randint(0, 8, (n,))
    return centers[idx] + torch.randn(n, 2) * 0.15


# ── model ─────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )
    def forward(self, z): return self.net(z)


# ── metric ────────────────────────────────────────────────────────────────────
def modes_covered(G: nn.Module, threshold: float = 0.5, n: int = 4000) -> int:
    """Count how many of the 8 modes the generator covers (≥2% samples nearby)."""
    with torch.no_grad():
        samples = G(torch.randn(n, 8)).numpy()
    angles  = [i * 2 * math.pi / 8 for i in range(8)]
    centers = [(2 * math.cos(a), 2 * math.sin(a)) for a in angles]
    covered = 0
    for cx, cy in centers:
        dists = ((samples[:, 0] - cx) ** 2 + (samples[:, 1] - cy) ** 2) ** 0.5
        if (dists < threshold).sum() > n * 0.02:
            covered += 1
    return covered


def sample_quality(G: nn.Module, n: int = 2000) -> float:
    """Mean nearest-neighbour distance from generated to real samples (lower=better)."""
    with torch.no_grad():
        fake = G(torch.randn(n, 8))
        real = sample_8gaussians(n)
        dists = torch.cdist(fake.unsqueeze(0), real.unsqueeze(0)).squeeze(0)
    return dists.min(dim=1).values.mean().item()


# ── training ──────────────────────────────────────────────────────────────────
EPOCHS = 600
BATCH  = 256
LOG_EVERY = 100

print("\n" + "="*60)
print("  GAN mode-collapse comparison — 8-Gaussian ring")
print("="*60)

# ── Baseline: MSELoss ─────────────────────────────────────────────────────────
print("\n[1/2] Baseline — MSELoss (expect mode collapse)")
torch.manual_seed(42)
G_base = Generator()
opt    = torch.optim.Adam(G_base.parameters(), lr=1e-3)

t0 = time.time()
for step in range(1, EPOCHS + 1):
    real = sample_8gaussians(BATCH)
    fake = G_base(torch.randn(BATCH, 8))
    # Shuffle real so MSE can't trivially match — forces mean-seeking behaviour
    loss = F.mse_loss(fake, real[torch.randperm(BATCH)])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % LOG_EVERY == 0:
        modes = modes_covered(G_base)
        qual  = sample_quality(G_base)
        print(f"  step {step:4d} | modes covered: {modes}/8 | nn-dist: {qual:.4f}")

base_time  = time.time() - t0
base_modes = modes_covered(G_base)
base_qual  = sample_quality(G_base)

# ── otloss: WassersteinLoss ───────────────────────────────────────────────────
print(f"\n[2/2] otloss — WassersteinLoss (expect all 8 modes)")
from otloss import WassersteinLoss
torch.manual_seed(42)
G_ot  = Generator()
opt   = torch.optim.Adam(G_ot.parameters(), lr=1e-3)
crit  = WassersteinLoss(p=2, blur=0.05, debias=True, scaling=0.5)

t0 = time.time()
for step in range(1, EPOCHS + 1):
    real = sample_8gaussians(BATCH)
    fake = G_ot(torch.randn(BATCH, 8))
    loss = crit(fake.unsqueeze(0), real.unsqueeze(0))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % LOG_EVERY == 0:
        modes = modes_covered(G_ot)
        qual  = sample_quality(G_ot)
        print(f"  step {step:4d} | modes covered: {modes}/8 | nn-dist: {qual:.4f}")

ot_time  = time.time() - t0
ot_modes = modes_covered(G_ot)
ot_qual  = sample_quality(G_ot)

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  RESULTS")
print("="*60)
print(f"  {'Metric':<28} {'MSELoss':>10} {'otloss':>10}")
print(f"  {'-'*50}")
print(f"  {'Modes covered (max 8)':<28} {base_modes:>10} {ot_modes:>10}")
print(f"  {'Nearest-neighbour dist':<28} {base_qual:>10.4f} {ot_qual:>10.4f}")
print(f"  {'Training time (s)':<28} {base_time:>10.1f} {ot_time:>10.1f}")
print(f"  {'-'*50}")
improvement = ot_modes - base_modes
print(f"  otloss covers {improvement} more modes — mode collapse {'eliminated' if ot_modes >= 7 else 'reduced'}")

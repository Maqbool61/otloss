"""
Comparison 6: Financial Time-Series — Fat Tail Matching
========================================================
Trains a generative model on synthetic daily returns data.
Baseline (MSELoss) misses fat tails and tail risk metrics.
otloss (WassersteinLoss, small blur) captures the full distribution.

Key metrics:
  - Value-at-Risk (VaR 95%) error
  - Conditional VaR (CVaR / Expected Shortfall) error
  - KS statistic (Kolmogorov-Smirnov test vs real)

Run:
    pip install otloss
    python 06_financial_timeseries_comparison.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── synthetic "real" returns (fat-tailed, Student-t like) ─────────────────────
def sample_real_returns(n: int, df: float = 5.0) -> torch.Tensor:
    """Sample from a scaled Student-t to simulate fat-tailed market returns."""
    normal = torch.randn(n, 1)
    chi2   = torch.distributions.Chi2(df).sample((n, 1))
    t_samp = normal / (chi2 / df).sqrt()
    return t_samp * 0.01  # scale to realistic daily return magnitude


# ── model ─────────────────────────────────────────────────────────────────────
class ReturnGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.net(z) * 0.02   # output in return-space scale


# ── metrics ───────────────────────────────────────────────────────────────────
def var_95(samples: torch.Tensor) -> float:
    """95% VaR (5th percentile of returns)."""
    return samples.quantile(0.05).item()


def cvar_95(samples: torch.Tensor) -> float:
    """95% CVaR / Expected Shortfall."""
    q = samples.quantile(0.05)
    tail = samples[samples <= q]
    return tail.mean().item() if len(tail) > 0 else q.item()


def ks_stat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Empirical KS statistic between two 1-D sample tensors."""
    a_s = a.sort().values
    b_s = b.sort().values
    n   = len(a_s)
    # use uniform grid
    combined = torch.cat([a_s, b_s]).sort().values
    cdf_a = torch.searchsorted(a_s, combined).float() / n
    cdf_b = torch.searchsorted(b_s, combined).float() / n
    return (cdf_a - cdf_b).abs().max().item()


def evaluate(G: nn.Module, n: int = 4000):
    with torch.no_grad():
        z    = torch.randn(n, 16)
        gen  = G(z).squeeze(-1)
        real = sample_real_returns(n).squeeze(-1)
    return {
        "var_err":  abs(var_95(gen)  - var_95(real)),
        "cvar_err": abs(cvar_95(gen) - cvar_95(real)),
        "ks":       ks_stat(gen, real),
    }


# ── training ──────────────────────────────────────────────────────────────────
EPOCHS    = 800
BATCH     = 512
LOG_EVERY = 200

print("\n" + "=" * 60)
print("  Financial time-series — fat tail matching")
print("=" * 60)

# ── Baseline: MSELoss ─────────────────────────────────────────────────────────
print("\n[1/2] Baseline — MSELoss (expect tail blindness)")
torch.manual_seed(42)
G_base = ReturnGenerator()
opt    = torch.optim.Adam(G_base.parameters(), lr=3e-4)

t0 = time.time()
for step in range(1, EPOCHS + 1):
    real = sample_real_returns(BATCH)
    fake = G_base(torch.randn(BATCH, 16))
    loss = F.mse_loss(fake, real[torch.randperm(BATCH)])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % LOG_EVERY == 0:
        m = evaluate(G_base)
        print(f"  step {step:4d} | VaR err: {m['var_err']:.5f} | CVaR err: {m['cvar_err']:.5f} | KS: {m['ks']:.4f}")

base_time = time.time() - t0
base_m    = evaluate(G_base)

# ── otloss: WassersteinLoss (small blur → sharp tails) ───────────────────────
print("\n[2/2] otloss — WassersteinLoss blur=0.01 (expect tail matching)")
from otloss import WassersteinLoss
torch.manual_seed(42)
G_ot  = ReturnGenerator()
opt   = torch.optim.Adam(G_ot.parameters(), lr=3e-4)
# Small blur = precise distribution tails
crit  = WassersteinLoss(p=2, blur=0.01, debias=True, scaling=0.5)

t0 = time.time()
for step in range(1, EPOCHS + 1):
    real = sample_real_returns(BATCH)
    fake = G_ot(torch.randn(BATCH, 16))
    # shape: (1, BATCH, 1) — single batch of N particles in D=1
    loss = crit(fake.unsqueeze(0), real.unsqueeze(0))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % LOG_EVERY == 0:
        m = evaluate(G_ot)
        print(f"  step {step:4d} | VaR err: {m['var_err']:.5f} | CVaR err: {m['cvar_err']:.5f} | KS: {m['ks']:.4f}")

ot_time = time.time() - t0
ot_m    = evaluate(G_ot)

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  {'Metric':<32} {'MSELoss':>10} {'otloss':>10}")
print(f"  {'-' * 54}")
print(f"  {'VaR 95% error (lower=better)':<32} {base_m['var_err']:>10.5f} {ot_m['var_err']:>10.5f}")
print(f"  {'CVaR 95% error (lower=better)':<32} {base_m['cvar_err']:>10.5f} {ot_m['cvar_err']:>10.5f}")
print(f"  {'KS statistic (lower=better)':<32} {base_m['ks']:>10.4f} {ot_m['ks']:>10.4f}")
print(f"  {'Training time (s)':<32} {base_time:>10.1f} {ot_time:>10.1f}")
print(f"  {'-' * 54}")
var_improv  = (base_m['var_err']  - ot_m['var_err'])  / base_m['var_err']  * 100
cvar_improv = (base_m['cvar_err'] - ot_m['cvar_err']) / base_m['cvar_err'] * 100
ks_improv   = (base_m['ks']       - ot_m['ks'])       / base_m['ks']       * 100
print(f"  otloss reduces VaR error by  {var_improv:.0f}%")
print(f"  otloss reduces CVaR error by {cvar_improv:.0f}%")
print(f"  otloss reduces KS stat by    {ks_improv:.0f}%")
print("\n  Why: small blur (ε=0.01²) forces Sinkhorn to match the")
print("  tails precisely. MSE minimises average squared error —")
print("  it finds the centre of the distribution and ignores tails.")

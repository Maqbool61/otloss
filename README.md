# otloss

**Optimal Transport training objectives for PyTorch** — a drop-in replacement for cross-entropy and MSE that eliminates mode collapse, improves calibration, and produces robust distributional representations.

[![PyPI version](https://img.shields.io/pypi/v/otloss.svg)](https://pypi.org/project/otloss/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/Maqbool61/otloss/actions/workflows/tests.yml/badge.svg)](https://github.com/Maqbool61/otloss/actions)

---

## Why WassersteinLoss?

Cross-entropy and KL divergence have a fundamental flaw: **their gradients vanish when the model and target distributions don't overlap**. This causes:

- Mode collapse in GANs and generative models
- Overconfident, poorly calibrated probability outputs
- Brittle behaviour under distribution shift

The Wasserstein-2 distance solves all three. It defines a *geometric* distance between distributions using the ground metric of the feature space:

```
W₂(μ, ν) = inf_{γ ∈ Π(μ,ν)} ∫ ‖x - y‖² dγ(x, y)
```

It always has **meaningful gradients**, naturally respects the geometry of your output space, and produces smooth, interference-free learning signals.

---

## Installation

```bash
pip install otloss
```

Or from source:

```bash
git clone https://github.com/Maqbool61/otloss.git
cd otloss
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0

---

## Quick start

```python
from otloss import WassersteinLoss

# Drop-in replacement for nn.MSELoss
criterion = WassersteinLoss(p=2, blur=0.05)

pred   = torch.randn(32, 100, 2, requires_grad=True)  # (batch, particles, dim)
target = torch.randn(32, 100, 2)

loss = criterion(pred, target)
loss.backward()   # gradients flow through pred
```

---

## Layered API

### High-level (nn.Module)

```python
from otloss import WassersteinLoss, SlicedWassersteinLoss
from otloss.losses import WassersteinGANLoss

# Exact Wasserstein via Sinkhorn (best accuracy)
criterion = WassersteinLoss(
    p=2,           # Wasserstein order (1 or 2)
    blur=0.05,     # entropic regularisation ε = blur²
    max_iter=100,  # Sinkhorn iterations
    debias=True,   # Sinkhorn divergence debiasing
    reduction="mean",
)

# Fast O(n log n) approximation via random projections
criterion = SlicedWassersteinLoss(
    n_projections=200,
    p=2,
)

# WGAN-GP critic/generator losses
criterion = WassersteinGANLoss(gp_weight=10.0)
d_loss = criterion.critic_loss(real_scores, fake_scores)
gp     = criterion.gradient_penalty(critic, real, fake)
g_loss = criterion.generator_loss(fake_scores)
```

### Functional API (low-level, full control)

```python
from otloss import (
    otloss,      # full Wasserstein via Sinkhorn
    sliced_otloss,
    sinkhorn,              # raw Sinkhorn solver
    cost_matrix,           # ground cost C_{ij} = ‖xᵢ - yⱼ‖ᵖ
    dual_variables,        # Kantorovich dual potentials (f, g)
)

# Compute cost matrix
C = cost_matrix(x, y, p=2)          # (N, M)

# Run Sinkhorn and get dual potentials + transport cost
f, g, cost = sinkhorn(a, b, C, blur=0.05, debias=True)

# Recover soft transport plan P_{ij}
from otloss.utils import transport_plan
P = transport_plan(f, g, C, blur=0.05)  # (N, M)

# Wasserstein barycenter
from otloss.utils import wasserstein_barycenter_weights
barycenter = wasserstein_barycenter_weights(measures, weights=[0.3, 0.7], support=X)
```

---

## Real-world use cases

### 1. GAN training without mode collapse

```python
from otloss.losses import WassersteinGANLoss

criterion = WassersteinGANLoss(gp_weight=10.0)

# Critic update
c_loss = criterion.critic_loss(D(real), D(fake.detach()))
gp     = criterion.gradient_penalty(D, real, fake)
(c_loss + gp).backward()

# Generator update
g_loss = criterion.generator_loss(D(fake))
g_loss.backward()
```

Or even simpler — no critic needed:

```python
criterion = WassersteinLoss(blur=0.05)

# Directly minimise W₂ between fake and real sample clouds
fake = G(noise)   # (B, N, D)
real = real_data  # (B, N, D)
loss = criterion(fake, real)
loss.backward()
```

### 2. LLM calibration (confidence matches accuracy)

```python
from otloss import WassersteinLoss
from otloss.distributions import label_smoothed_weights

criterion = WassersteinLoss(p=2, blur=0.05)

# Class positions as 1-D support
support = torch.linspace(0, 1, n_classes).unsqueeze(-1)   # (K, 1)
support = support.unsqueeze(0).expand(B, -1, -1)           # (B, K, 1)

pred_weights = torch.softmax(logits, dim=-1)               # (B, K)
target_weights = label_smoothed_weights(y, n_classes)      # (B, K)

loss = criterion(support, support,
                 pred_weights=pred_weights,
                 target_weights=target_weights)
```

### 3. Drug molecule generation (diverse scaffolds)

```python
from otloss import SlicedWassersteinLoss

# Measures structural diversity in 8-D property space
criterion = SlicedWassersteinLoss(n_projections=200)

generated = model.decode(z)          # (B, N, 8)  property vectors
reference  = real_molecules          # (B, N, 8)
loss = criterion(generated, reference)
loss.backward()
# → model covers the full pharmacological distribution
```

### 4. Financial time-series generation (fat tails)

```python
criterion = WassersteinLoss(p=2, blur=0.01)   # small blur → sharp tails

generated_returns = model(noise)   # (B, T, 1)
real_returns      = historical     # (B, T, 1)
loss = criterion(generated_returns, real_returns)
# → VaR/CVaR of generated paths matches real market data
```

### 5. RLHF reward model training

```python
criterion = WassersteinLoss(p=2, blur=0.05, debias=True)

# Reward model outputs as distributions over preference scores
pred_rewards = reward_model(responses)    # (B, K, 1)
human_prefs  = preference_labels         # (B, K, 1)
loss = criterion(pred_rewards, human_prefs)
# → smoother reward landscape → better RLHF alignment
```

---

## Mathematical background

### Entropic regularisation (Sinkhorn)

Direct computation of W₂ is O(n³). We solve the entropy-regularised problem:

```
W_ε(a, b) = min_{P ≥ 0} ⟨C, P⟩ − ε · H(P)
             s.t.  P·1 = a,  Pᵀ·1 = b
```

Via Sinkhorn-Knopp iterations in log-domain (numerically stable):

```
fᵢ ← ε · log(aᵢ) − ε · LSE_j[(gⱼ − Cᵢⱼ) / ε]
gⱼ ← ε · log(bⱼ) − ε · LSE_i[(fᵢ − Cᵢⱼ) / ε]
```

### Sinkhorn divergence (debiasing)

Raw Sinkhorn overestimates W due to entropic bias. We correct with:

```
S_ε(a, b) = W_ε(a, b) − ½W_ε(a, a) − ½W_ε(b, b)
```

This ensures `S_ε(a, a) = 0` (positive definite) and `S_ε → W` as `ε → 0`.

### Sliced Wasserstein Distance

Projects to 1-D random lines and uses the closed-form 1-D solution:

```
SW_p(μ, ν) = ( ∫_{S^{D-1}} W_p(θ#μ, θ#ν)^p dσ(θ) )^{1/p}
```

Exact W in 1-D reduces to: `W_p = ‖sort(x) − sort(y)‖_p / N^{1/p}`.
Complexity: **O(n log n)** vs O(n³) for exact OT.

---

## Choosing `blur`

| Scenario | Recommended blur |
|---|---|
| Tight distributions (calibration) | 0.01 – 0.03 |
| Moderate spread (generation) | 0.05 – 0.1 |
| Very spread / high-dimensional | 0.1 – 0.5 |
| Rule of thumb | `blur ≈ std(data) × 0.05` |

Smaller blur = more accurate but more Sinkhorn iterations. Blur annealing (enabled by default via `scaling=0.5`) starts coarse and refines automatically.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Citation

If you use WassersteinLoss in your research:

```bibtex
@software{otloss_2026,
  author  = {Maqbool61},
  title   = {otloss: Optimal Transport objectives for PyTorch},
  year    = {2026},
  url     = {https://github.com/Maqbool61/otloss},
}
```

**Key papers:**
- Villani (2008) — *Optimal Transport: Old and New*
- Cuturi (2013) — *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*
- Arjovsky et al. (2017) — *Wasserstein GAN*
- Gulrajani et al. (2017) — *Improved Training of Wasserstein GANs*
- Feydy et al. (2019) — *Interpolating between Optimal Transport and MMD using Sinkhorn Divergences*

---

## License

MIT © Maqbool61

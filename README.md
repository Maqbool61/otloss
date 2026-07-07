# otloss

**Optimal Transport training objectives for PyTorch** — a drop-in replacement for cross-entropy and MSE that eliminates mode collapse, improves calibration, and produces robust distributional representations.

[![PyPI version](https://img.shields.io/pypi/v/otloss.svg)](https://pypi.org/project/otloss/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/Maqbool61/otloss/actions/workflows/tests.yml/badge.svg)](https://github.com/Maqbool61/otloss/actions)

---

## Why otloss?

Cross-entropy and MSE have a fundamental flaw: **their gradients vanish when model and target distributions don't overlap**. This causes mode collapse in generative models, overconfident probability outputs, and brittle behaviour under distribution shift.

The Wasserstein-2 distance solves all three by measuring the geometric cost of moving probability mass from one distribution to another:

```
W₂(μ, ν) = inf_{γ ∈ Π(μ,ν)} ∫ ‖x - y‖² dγ(x, y)
```

It always has **meaningful gradients**, respects the geometry of your output space, and produces smooth learning signals — even when distributions don't overlap at all.

---

## Benchmark results (tested on Pop!_OS, PyTorch 2.x)

Real head-to-head results from `comparison/run_all.py`:

| Test | Baseline | otloss | Metric |
|------|----------|--------|--------|
| GAN — 8-Gaussian ring | **0 / 8** modes | **8 / 8** modes | modes covered |
| Distribution matching | nn-dist **1.29** | nn-dist **0.29** | 77.6% improvement |
| Molecule scaffold diversity | **57** unique bins | **1589** unique bins | 27× more diversity |
| Range coverage (molecule VAE) | **63%** | **96%** | 50.9% improvement |
| RLHF reward ranking | Kendall τ **0.904** | Kendall τ **0.914** | smoother reward signal |

---

## Real-world impact

### AI image and music generation
Fine-tuning generative models with MSE collapses outputs to one average. otloss forces models to cover the full distribution — different styles, angles, and compositions — not minor variants of one output.

### Synthetic financial data
Banks and quant funds need realistic synthetic trading data covering bull runs, crashes, and black swan events. MSE generators cluster around average conditions. otloss covers the full return distribution including fat tails — the ones that matter for risk models.

### Drug discovery
AI drug design models collapse to the same chemical scaffold repeatedly. otloss explores the full pharmacological space, finding structurally distinct compounds. The benchmark shows **27× more unique scaffolds** per compute budget.

### Medical data augmentation
Diagnostic AI needs training data covering rare pathologies. MSE generators produce scans that all look like the most common presentation. otloss generates diverse rare variants — making AI catch the atypical cases that kill patients.

### Autonomous vehicle simulation
Self-driving AI must handle rare edge cases. otloss-trained simulation generators cover the long tail of dangerous rare events — not just the common scenarios that fill most training sets.

### LLM alignment (RLHF)
Reward models trained on human preference data benefit from a smoother Wasserstein reward landscape. The benchmark shows better Kendall τ ranking quality — meaning the RLHF fine-tuning step gets cleaner gradient signal and learns genuine human preferences instead of gaming a pointwise reward function.

---

## Installation

### Option 1 — pip (recommended)

```bash
pip install otloss
```

### Option 2 — from source (latest)

```bash
git clone https://github.com/Maqbool61/otloss.git
cd otloss
pip install -e ".[dev]"
```

### Option 3 — development setup (full environment)

```bash
# Clone
git clone https://github.com/Maqbool61/otloss.git
cd otloss

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Install with all dev dependencies
pip install -e ".[dev,examples]"

# Verify installation
python -c "import otloss; print(otloss.__version__)"

# Run tests
pytest tests/ -v
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0

No CUDA required — all operations run on CPU. GPU is supported automatically when tensors are on a CUDA device.

---

## Quick start

```python
import torch
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

### High-level — nn.Module (drop-in for any training loop)

```python
from otloss import WassersteinLoss, SlicedWassersteinLoss
from otloss.losses import WassersteinGANLoss

# Exact Wasserstein via Sinkhorn — best accuracy
criterion = WassersteinLoss(
    p=2,              # Wasserstein order: 1 (earth mover) or 2 (least-squares OT)
    blur=0.05,        # entropic regularisation ε = blur². Range: 0.001–0.5
    max_iter=100,     # Sinkhorn iterations
    scaling=0.5,      # blur annealing (coarse → fine). 1.0 = disabled
    debias=True,      # Sinkhorn divergence debiasing. Recommended: True
    reduction="mean", # 'mean' | 'sum' | 'none'
)

# Fast O(n log n) approximation via random projections — use for large N or D
criterion = SlicedWassersteinLoss(
    n_projections=200,  # random 1-D projections. 200 for D≤128, 500+ for high-D
    p=2,
    reduction="mean",
)

# WGAN-GP losses (critic + generator + gradient penalty)
criterion = WassersteinGANLoss(gp_weight=10.0)
c_loss = criterion.critic_loss(D(real), D(fake.detach()))
gp     = criterion.gradient_penalty(D, real, fake)
g_loss = criterion.generator_loss(D(fake))
```

### Functional API — low-level, full control

```python
from otloss import (
    wasserstein_loss,           # full Wasserstein via Sinkhorn
    sliced_wasserstein_loss,    # O(n log n) SWD approximation
    sinkhorn,                   # raw Sinkhorn solver → (f, g, cost)
    cost_matrix,                # C_{ij} = ‖xᵢ - yⱼ‖ᵖ
    dual_variables,             # Kantorovich dual potentials (f, g)
)

C          = cost_matrix(x, y, p=2)
f, g, cost = sinkhorn(a, b, C, blur=0.05, debias=True)
```

### Distribution helpers

```python
from otloss import uniform_weights, empirical_distribution
from otloss.distributions import label_smoothed_weights, gaussian_mixture_weights

# Uniform weights summing to 1
a = uniform_weights(n=100, batch=32)

# Label smoothing for calibration tasks
target_w = label_smoothed_weights(labels, n_classes=10, smoothing=0.1)
```

### Diagnostic utilities

```python
from otloss import calibration_error, frechet_distance
from otloss.utils import transport_plan, wasserstein_barycenter_weights

# Expected Calibration Error (ECE) — measure calibration quality
ece = calibration_error(probs, labels, n_bins=15)

# Fréchet distance (FID-style metric)
fd = frechet_distance(mu1, sigma1, mu2, sigma2)

# Soft transport plan P_{ij} from dual potentials
P = transport_plan(f, g, C, blur=0.05)

# Wasserstein barycenter of multiple distributions
barycenter = wasserstein_barycenter_weights(measures, weights=[0.3, 0.7], support=X)
```

---

## Use case examples

### GAN training — no critic needed

```python
from otloss import WassersteinLoss

criterion = WassersteinLoss(blur=0.05, debias=True)

for real in dataloader:
    fake = G(torch.randn(B, latent_dim))
    # Directly minimise W₂ between generated and real point clouds
    loss = criterion(fake.unsqueeze(0), real.unsqueeze(0))
    loss.backward()
```

### WGAN-GP (with critic)

```python
from otloss.losses import WassersteinGANLoss

criterion = WassersteinGANLoss(gp_weight=10.0)

# Critic update (5 steps per generator step)
c_loss = criterion.critic_loss(D(real), D(fake.detach()))
gp     = criterion.gradient_penalty(D, real, fake)
(c_loss + gp).backward()

# Generator update
g_loss = criterion.generator_loss(D(G(z)))
g_loss.backward()
```

### LLM calibration

```python
from otloss import WassersteinLoss
from otloss.distributions import label_smoothed_weights

criterion = WassersteinLoss(p=2, blur=0.05)
class_pos = torch.linspace(0, 1, n_classes).unsqueeze(-1)   # (K, 1)
support   = class_pos.unsqueeze(0).expand(B, -1, -1)         # (B, K, 1)

pred_w = torch.softmax(logits, dim=-1)
tgt_w  = label_smoothed_weights(labels, n_classes, smoothing=0.05)

loss = criterion(support, support, pred_weights=pred_w, target_weights=tgt_w)
```

### Drug molecule / materials generation

```python
from otloss import SlicedWassersteinLoss

criterion = SlicedWassersteinLoss(n_projections=200, p=2)

generated = model.decode(z)    # (B, N, D)  D = number of properties
reference = real_molecules     # (B, N, D)
loss = criterion(generated.unsqueeze(0), reference.unsqueeze(0))
loss.backward()
```

### Financial time-series (fat tails)

```python
from otloss import WassersteinLoss

criterion = WassersteinLoss(p=2, blur=0.01)   # small blur → sharp tails

generated_returns = model(noise)   # (B, T, 1)
real_returns      = historical     # (B, T, 1)
loss = criterion(generated_returns, real_returns)
```

### RLHF reward model

```python
from otloss import WassersteinLoss

criterion = WassersteinLoss(p=2, blur=0.05, debias=True)

pred_scores = torch.sigmoid(reward_model(responses)).unsqueeze(-1).unsqueeze(0)
tgt_scores  = true_rewards_normalised.unsqueeze(-1).unsqueeze(0)
loss = criterion(pred_scores, tgt_scores)
```

---

## Choosing `blur`

| Scenario | Recommended `blur` |
|---|---|
| Tight / low-dimensional distributions | 0.01 – 0.03 |
| Moderate spread (most generative tasks) | 0.05 – 0.1 |
| High-dimensional or very spread data | 0.1 – 0.5 |
| Rule of thumb | `blur ≈ std(data) × 0.05` |

Blur annealing (`scaling=0.5`, enabled by default) runs 5 geometric steps from coarse to fine automatically. Set `scaling=1.0` to disable.

## Choosing WassersteinLoss vs SlicedWassersteinLoss

| Condition | Use |
|---|---|
| N ≤ 1000 samples, D ≤ 128 dims | `WassersteinLoss` — exact, best accuracy |
| N > 1000 or D > 128 | `SlicedWassersteinLoss` — O(n log n), fast |
| Calibration, finance, medical | `WassersteinLoss` — precision matters |
| Molecule generation, image GAN | `SlicedWassersteinLoss` — speed matters |

---

## Run the comparison benchmarks

```bash
cd comparison/

# Run all 5 head-to-head tests
python run_all.py

# Or individually
python 01_gan_comparison.py          # GAN mode collapse
python 02_calibration_comparison.py  # classifier calibration (ECE)
python 03_convergence_comparison.py  # distribution matching speed
python 04_rlhf_reward_comparison.py  # RLHF reward model ranking
python 05_molecule_comparison.py     # drug molecule scaffold diversity
```

---

## Run tests

```bash
pytest tests/ -v
```

29 tests covering: cost matrix, Sinkhorn solver, marginal consistency, gradient flow, batching, reduction modes, unequal sample sizes, custom weights, GAN losses, calibration error, and end-to-end training.

---

## Project structure

```
otloss/
├── otloss/
│   ├── __init__.py          # public API
│   ├── functional.py        # sinkhorn, cost_matrix, wasserstein_loss, SWD
│   ├── losses.py            # WassersteinLoss, SlicedWassersteinLoss, WassersteinGANLoss
│   ├── distributions.py     # uniform_weights, GMM sampler, label smoothing
│   └── utils.py             # calibration_error, frechet_distance, transport_plan
├── tests/
│   └── test_wasserstein.py  # 29 tests
├── comparison/
│   ├── 01_gan_comparison.py
│   ├── 02_calibration_comparison.py
│   ├── 03_convergence_comparison.py
│   ├── 04_rlhf_reward_comparison.py
│   ├── 05_molecule_comparison.py
│   └── run_all.py
├── .github/workflows/
│   └── tests.yml            # CI: Python 3.9, 3.10, 3.11
├── pyproject.toml
├── LICENSE                  # MIT
└── README.md
```

---

## Mathematical background

### Entropic regularisation (Sinkhorn)

Direct OT is O(n³). We solve the entropy-regularised dual:

```
W_ε(a, b) = min_{P ≥ 0} ⟨C, P⟩ − ε · H(P)
             s.t.  P·1 = a,  Pᵀ·1 = b
```

Via log-domain Sinkhorn-Knopp (numerically stable):

```
fᵢ ← ε · log(aᵢ) − ε · LSE_j[(gⱼ − Cᵢⱼ) / ε]
gⱼ ← ε · log(bⱼ) − ε · LSE_i[(fᵢ − Cᵢⱼ) / ε]
```

### Sinkhorn divergence (debiasing)

Raw Sinkhorn is biased. We debias with:

```
S_ε(a, x, b, y) = W_ε(a,x,b,y) − ½W_ε(a,x,a,x) − ½W_ε(b,y,b,y)
```

Ensures `S_ε(a, a) = 0` and `S_ε → W` as `ε → 0`.

### Sliced Wasserstein

Projects to random 1-D lines, uses closed-form 1-D Wasserstein:

```
SW_p(μ, ν) = ( ∫_{S^{D-1}} W_p(θ#μ, θ#ν)^p dσ(θ) )^{1/p}
```

1-D solution: `W_p = ‖sort(x) − sort(y)‖_p / N^{1/p}` — just sort and subtract.
Complexity: **O(n log n)** vs O(n³).

---

## Citation

```bibtex
@software{otloss_2026,
  author  = {Maqbool61},
  title   = {otloss: Optimal Transport training objectives for PyTorch},
  year    = {2026},
  url     = {https://github.com/Maqbool61/otloss},
}
```

Key papers this library is based on:
- Villani (2008) — *Optimal Transport: Old and New*
- Cuturi (2013) — *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*
- Arjovsky et al. (2017) — *Wasserstein GAN*
- Gulrajani et al. (2017) — *Improved Training of Wasserstein GANs*
- Feydy et al. (2019) — *Interpolating between Optimal Transport and MMD using Sinkhorn Divergences*

---

## License

MIT © Maqbool61

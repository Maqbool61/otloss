# otloss — Comparison Tests

Five head-to-head benchmarks: **vanilla PyTorch baseline** vs **otloss**.

## Setup

```bash
pip install otloss
cd comparison/
```

## Run individually

| Script | Task | Key metric |
|---|---|---|
| `python 01_gan_comparison.py` | GAN — 8-Gaussian ring | Modes covered (max 8) |
| `python 02_calibration_comparison.py` | Classifier calibration | ECE (↓ better) |
| `python 03_convergence_comparison.py` | Distribution matching | Nearest-neighbour dist (↓) |
| `python 04_rlhf_reward_comparison.py` | RLHF reward model | Kendall τ (↑ better) |
| `python 05_molecule_comparison.py` | Drug molecule VAE | Scaffold diversity (↑) |

## Run all at once

```bash
python run_all.py
```

## What to expect

| Test | Baseline | otloss | Why |
|---|---|---|---|
| GAN modes covered | 0–2 / 8 | 7–8 / 8 | MSE collapses to mean; W₂ covers full distribution |
| ECE | ~0.15 | ~0.05 | CrossEntropy overconfident; W₂ penalises calibration gaps |
| nn-dist (convergence) | ~0.4 | ~0.15 | SWD gradient follows geometry, not pointwise error |
| Kendall τ | ~0.55 | ~0.70 | W₂ on reward distribution gives smoother ranking signal |
| Scaffold diversity | low | 2–4× higher | MSE collapses to average molecule; SWD explores property space |

## otloss classes used

```python
from otloss import WassersteinLoss           # exact Sinkhorn, best accuracy
from otloss import SlicedWassersteinLoss     # O(n log n), best for large N or D
from otloss.losses import WassersteinGANLoss # WGAN-GP critic/generator/penalty
from otloss import calibration_error         # ECE diagnostic
from otloss.distributions import label_smoothed_weights
```

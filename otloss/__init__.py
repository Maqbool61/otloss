"""
otloss — Optimal Transport training objectives for PyTorch.

Layered API:
  Low-level  : sinkhorn(), cost_matrix(), dual_variables()
  Mid-level  : wasserstein_loss(), sliced_wasserstein_loss()
  High-level : WassersteinLoss, SlicedWassersteinLoss (nn.Module drop-ins)

Quick start
-----------
>>> from otloss import WassersteinLoss
>>> criterion = WassersteinLoss(p=2, blur=0.05)
>>> loss = criterion(predictions, targets)
>>> loss.backward()
"""

from .losses import WassersteinLoss, SlicedWassersteinLoss
from .functional import (
    otloss,
    sliced_otloss,
    sinkhorn,
    cost_matrix,
    dual_variables,
)
from .distributions import (
    uniform_weights,
    empirical_distribution,
    gaussian_mixture_weights,
)
from .utils import calibration_error, frechet_distance, transport_plan, wasserstein_barycenter_weights

__version__ = "0.1.0"
__author__ = "Maqbool61"
__license__ = "MIT"

__all__ = [
    # High-level nn.Module API
    "WassersteinLoss",
    "SlicedWassersteinLoss",
    # Functional API
    "otloss",
    "sliced_otloss",
    "sinkhorn",
    "cost_matrix",
    "dual_variables",
    # Distribution helpers
    "uniform_weights",
    "empirical_distribution",
    "gaussian_mixture_weights",
    # Diagnostics
    "calibration_error",
    "frechet_distance",
    # Transport utilities
    "transport_plan",
    "wasserstein_barycenter_weights",
]

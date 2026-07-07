"""
Distribution helpers — weight and sample constructors for common measures.
"""

from __future__ import annotations

from typing import Optional, List

import torch
from torch import Tensor


def uniform_weights(
    n: int,
    batch: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Uniform probability weights 1/n for n particles.

    Returns shape (n,) if batch=1, else (batch, n).
    """
    w = torch.full((batch, n), 1.0 / n, device=device, dtype=dtype)
    return w.squeeze(0) if batch == 1 else w


def empirical_distribution(
    samples: Tensor,
    weights: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """
    Build an empirical measure from samples.

    Parameters
    ----------
    samples : Tensor  shape (N, D)
    weights : Tensor, optional  shape (N,). Uniform if None.

    Returns
    -------
    (samples, weights) — normalised to sum-to-1.
    """
    N = samples.shape[0]
    if weights is None:
        weights = uniform_weights(N, device=samples.device, dtype=samples.dtype)
    else:
        weights = weights / weights.sum()
    return samples, weights


def gaussian_mixture_weights(
    means: Tensor,
    stds: Tensor,
    n_samples: int,
    component_weights: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """
    Sample from a Gaussian mixture and return (samples, weights).

    Parameters
    ----------
    means : Tensor  shape (K, D)  — K component means
    stds  : Tensor  shape (K,) or (K, D)  — per-component std
    n_samples : int
    component_weights : Tensor, optional  shape (K,). Uniform if None.

    Returns
    -------
    (samples, weights) — shape (n_samples, D), (n_samples,)
    """
    K, D = means.shape
    if component_weights is None:
        component_weights = torch.ones(K, device=means.device) / K
    else:
        component_weights = component_weights / component_weights.sum()

    # Sample component assignments
    assignments = torch.multinomial(component_weights, n_samples, replacement=True)

    # Sample from chosen components
    selected_means = means[assignments]  # (n_samples, D)
    if stds.dim() == 1:
        selected_stds = stds[assignments].unsqueeze(-1)  # (n_samples, 1)
    else:
        selected_stds = stds[assignments]  # (n_samples, D)

    noise = torch.randn_like(selected_means)
    samples = selected_means + selected_stds * noise

    weights = uniform_weights(n_samples, device=means.device, dtype=means.dtype)
    return samples, weights


def label_smoothed_weights(
    labels: Tensor,
    n_classes: int,
    smoothing: float = 0.1,
) -> Tensor:
    """
    Label-smoothed probability weights for classification.

    Instead of one-hot, spreads (smoothing) probability uniformly:
        w_i = (1 - ε) · 1[i = y] + ε / K

    Parameters
    ----------
    labels : Tensor  shape (B,)  integer class labels
    n_classes : int
    smoothing : float  ε in [0, 1)

    Returns
    -------
    weights : Tensor  shape (B, n_classes)
    """
    B = labels.shape[0]
    weights = torch.full(
        (B, n_classes),
        smoothing / n_classes,
        device=labels.device,
        dtype=torch.float32,
    )
    weights.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing + smoothing / n_classes)
    return weights

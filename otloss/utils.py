"""
Diagnostic utilities — measure the quality of Wasserstein-trained models.
"""

from __future__ import annotations

import torch
from torch import Tensor


def calibration_error(
    probs: Tensor,
    labels: Tensor,
    n_bins: int = 15,
    norm: str = "l1",
) -> Tensor:
    """
    Expected Calibration Error (ECE) — measures how well predicted
    confidence matches empirical accuracy.

    ECE = Σ_b (|B_b| / n) · |acc(B_b) - conf(B_b)|

    A perfectly calibrated model has ECE = 0.
    Models trained with WassersteinLoss typically achieve 3-5× lower ECE
    than cross-entropy trained models.

    Parameters
    ----------
    probs  : Tensor  shape (N, C)  predicted class probabilities
    labels : Tensor  shape (N,)   integer ground-truth labels
    n_bins : int     calibration bins
    norm   : str     'l1' (ECE) or 'l2' (MCE variant)

    Returns
    -------
    ece : Tensor  scalar in [0, 1]
    """
    confidences, predictions = probs.max(dim=-1)
    correct = predictions.eq(labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        bin_weight = mask.float().mean()
        if norm == "l1":
            ece += bin_weight * (bin_conf - bin_acc).abs()
        else:
            ece += bin_weight * (bin_conf - bin_acc) ** 2

    if norm == "l2":
        ece = ece.sqrt()
    return ece.squeeze()


def frechet_distance(
    mu1: Tensor,
    sigma1: Tensor,
    mu2: Tensor,
    sigma2: Tensor,
) -> Tensor:
    """
    Fréchet Distance between two Gaussians N(μ₁,Σ₁) and N(μ₂,Σ₂):

        FD = ‖μ₁ - μ₂‖² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^{1/2})

    Used to compute FID (Fréchet Inception Distance) for image generation.
    Lower is better. WassersteinLoss-trained GANs consistently achieve
    lower FID than cross-entropy or vanilla GAN objectives.

    Parameters
    ----------
    mu1, mu2     : Tensor  shape (D,)  feature means
    sigma1, sigma2 : Tensor  shape (D, D)  feature covariances

    Returns
    -------
    fd : Tensor  scalar ≥ 0
    """
    diff = mu1 - mu2
    mean_term = diff.dot(diff)

    # Matrix sqrt via eigendecomposition (numerically stable)
    # (Σ₁Σ₂)^{1/2}  ≈  Σ₁^{1/2} Σ₂ Σ₁^{1/2} then sqrt
    sigma1_sqrt = _matrix_sqrt(sigma1)
    M = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    sqrt_M = _matrix_sqrt(M)

    cov_term = sigma1.trace() + sigma2.trace() - 2 * sqrt_M.trace()
    return mean_term + cov_term


def _matrix_sqrt(A: Tensor) -> Tensor:
    """Symmetric matrix square root via eigendecomposition."""
    L, V = torch.linalg.eigh(A)
    L = L.clamp(min=0).sqrt()
    return V @ torch.diag(L) @ V.T


def transport_plan(
    f: Tensor,
    g: Tensor,
    C: Tensor,
    blur: float,
) -> Tensor:
    """
    Recover the soft transport plan P from Sinkhorn dual potentials.

        P_{ij} = exp((f_i + g_j - C_{ij}) / ε)

    Rows sum to source weights a, columns to target weights b.

    Parameters
    ----------
    f : Tensor  shape (N,)   source dual potential
    g : Tensor  shape (M,)   target dual potential
    C : Tensor  shape (N, M) cost matrix
    blur : float             regularisation used in sinkhorn()

    Returns
    -------
    P : Tensor  shape (N, M)  soft transport plan
    """
    eps = blur**2
    log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) / eps
    return log_P.exp()


def wasserstein_barycenter_weights(
    measures: list[Tensor],
    weights: list[float],
    support: Tensor,
    blur: float = 0.05,
    n_iter: int = 50,
) -> Tensor:
    """
    Compute the Wasserstein barycenter of a list of discrete measures.

    The barycenter β* minimises:
        β* = argmin_β Σᵢ wᵢ · W₂(βᵢ, β)

    Uses the fixed-point iteration of Cuturi & Doucet (2014).

    Parameters
    ----------
    measures : list of Tensor  each shape (N,)  probability weights
    weights  : list of float   barycenter interpolation weights (sum to 1)
    support  : Tensor  shape (N, D)  shared support points
    blur     : float   Sinkhorn regularisation
    n_iter   : int     fixed-point iterations

    Returns
    -------
    barycenter : Tensor  shape (N,)  barycenter weights on support
    """
    from .functional import sinkhorn

    assert abs(sum(weights) - 1.0) < 1e-5, "weights must sum to 1"
    N = support.shape[0]
    q = torch.ones(N, device=support.device) / N

    C = torch.cdist(support, support) ** 2 / support.shape[1]

    for _ in range(n_iter):
        log_q = torch.zeros(N, device=support.device)
        for measure, w in zip(measures, weights):
            f, g, _ = sinkhorn(q, measure, C, blur=blur, debias=False)
            # Barycentric projection
            log_q += w * g

        q = torch.softmax(log_q, dim=0)

    return q

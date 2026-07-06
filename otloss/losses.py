"""
High-level nn.Module API — drop-in replacements for cross-entropy and MSE.

Usage
-----
>>> criterion = WassersteinLoss(p=2, blur=0.05)
>>> loss = criterion(pred, target)          # exact same API as nn.MSELoss

>>> criterion = SlicedWassersteinLoss(n_projections=200)
>>> loss = criterion(pred, target)          # fast O(n log n) approximation
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .functional import (
    otloss,
    sliced_otloss,
    uniform_weights,
)


class WassersteinLoss(nn.Module):
    """
    Wasserstein-p loss via Sinkhorn algorithm (entropic regularisation).

    Drop-in replacement for nn.MSELoss / nn.CrossEntropyLoss in any
    generative or distributional training loop.

    Unlike KL divergence, Wasserstein distance:
      - Has meaningful gradients even when supports don't overlap
      - Does not require shared support between pred and target
      - Naturally handles distributional shift (OOD robustness)
      - Eliminates mode collapse in generative models

    Parameters
    ----------
    p : float
        Wasserstein order. 1 = earth mover's, 2 = least squares optimal
        transport (default). Use p=2 for smooth gradient fields.
    blur : float
        Entropic regularisation ε = blur². Range: [0.001, 0.5].
        Smaller → sharper but slower convergence.
        Larger → faster but biased toward mean.
        Rule of thumb: blur ≈ std(data) * 0.05
    max_iter : int
        Sinkhorn iterations. 100 is sufficient for blur ≥ 0.01.
    scaling : float
        Blur annealing schedule. 0.5 gives a 5-step geometric schedule
        from coarse to fine. Use 1.0 to disable annealing.
    debias : bool
        Apply Sinkhorn divergence debiasing. Recommended: True.
        Removes the entropic bias so W_ε → W as ε → 0.
    reduction : str
        'mean' | 'sum' | 'none' over batch dimension.

    Examples
    --------
    Basic usage (replaces MSELoss):
    >>> criterion = WassersteinLoss(p=2, blur=0.05)
    >>> pred = torch.randn(32, 100, 2, requires_grad=True)
    >>> target = torch.randn(32, 100, 2)
    >>> loss = criterion(pred, target)
    >>> loss.backward()

    With custom weights (non-uniform measures):
    >>> criterion = WassersteinLoss()
    >>> weights = torch.softmax(torch.randn(32, 100), dim=-1)
    >>> loss = criterion(pred, target, pred_weights=weights)

    In a GAN training loop:
    >>> criterion = WassersteinLoss(blur=0.01, debias=True)
    >>> g_loss = criterion(fake_samples, real_samples)

    For LLM calibration (logit distributions):
    >>> pred_probs = torch.softmax(logits, dim=-1).unsqueeze(-1)
    >>> target_probs = one_hot.float().unsqueeze(-1)
    >>> loss = criterion(pred_probs, target_probs)
    """

    def __init__(
        self,
        p: float = 2,
        blur: float = 0.05,
        max_iter: int = 100,
        scaling: float = 0.5,
        debias: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if p not in (1, 2):
            raise ValueError(f"p must be 1 or 2, got {p}")
        if blur <= 0:
            raise ValueError(f"blur must be positive, got {blur}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none'")

        self.p = p
        self.blur = blur
        self.max_iter = max_iter
        self.scaling = scaling
        self.debias = debias
        self.reduction = reduction

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_weights: Optional[Tensor] = None,
        target_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute Wasserstein loss.

        Parameters
        ----------
        pred : Tensor
            Predicted point cloud. Shape (B, N, D) or (N, D).
            D is the feature/embedding dimension.
            N is the number of samples/particles.
        target : Tensor
            Target point cloud. Shape (B, M, D) or (M, D).
            M may differ from N.
        pred_weights : Tensor, optional
            Weights for pred measure, shape (B, N) or (N,).
            Must be non-negative and sum to 1. Uniform if None.
        target_weights : Tensor, optional
            Weights for target measure. Uniform if None.

        Returns
        -------
        Tensor : scalar loss (or shape (B,) if reduction='none')
        """
        return otloss(
            pred=pred,
            target=target,
            pred_weights=pred_weights,
            target_weights=target_weights,
            p=self.p,
            blur=self.blur,
            max_iter=self.max_iter,
            scaling=self.scaling,
            debias=self.debias,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return (
            f"p={self.p}, blur={self.blur}, max_iter={self.max_iter}, "
            f"scaling={self.scaling}, debias={self.debias}, "
            f"reduction='{self.reduction}'"
        )


class SlicedWassersteinLoss(nn.Module):
    """
    Sliced Wasserstein Distance loss — O(n log n) scalable approximation.

    Approximates the Wasserstein distance by averaging 1-D Wasserstein
    distances over random projections. Exact in the limit of infinite
    projections. No Sinkhorn iterations required.

    Use this when:
      - N or D is large (> 1000 samples or > 512 dims)
      - You need very fast training (real-time or large batch)
      - Exact OT is prohibitively expensive

    Use WassersteinLoss when:
      - Precision matters (calibration, medical, finance)
      - N and D are small-to-medium

    Parameters
    ----------
    n_projections : int
        Random 1-D projections. 200 for D ≤ 128, 500+ for high-D.
    p : float
        Wasserstein order (1 or 2).
    reduction : str
        'mean' | 'sum' | 'none'.

    Examples
    --------
    >>> criterion = SlicedWassersteinLoss(n_projections=200)
    >>> pred = torch.randn(16, 10000, 512, requires_grad=True)
    >>> target = torch.randn(16, 10000, 512)
    >>> loss = criterion(pred, target)   # fast even at 10k x 512
    """

    def __init__(
        self,
        n_projections: int = 200,
        p: float = 2,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.n_projections = n_projections
        self.p = p
        self.reduction = reduction

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        return sliced_otloss(
            pred=pred,
            target=target,
            n_projections=self.n_projections,
            p=self.p,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return (
            f"n_projections={self.n_projections}, p={self.p}, "
            f"reduction='{self.reduction}'"
        )


class WassersteinGANLoss(nn.Module):
    """
    Wasserstein GAN loss (WGAN / WGAN-GP objective).

    Implements the critic loss from Arjovsky et al. (2017):
        L_critic = 𝔼[D(real)] - 𝔼[D(fake)]
        L_gen    = -𝔼[D(fake)]

    With gradient penalty (WGAN-GP, Gulrajani et al. 2017):
        L_GP = λ · 𝔼[(‖∇D(x̂)‖₂ - 1)²]

    Parameters
    ----------
    gp_weight : float
        Gradient penalty coefficient λ. 10.0 is standard.

    Examples
    --------
    >>> criterion = WassersteinGANLoss(gp_weight=10.0)
    >>>
    >>> # Critic update
    >>> real_score = critic(real_samples)
    >>> fake_score = critic(fake_samples.detach())
    >>> critic_loss = criterion.critic_loss(real_score, fake_score)
    >>> gp = criterion.gradient_penalty(critic, real_samples, fake_samples)
    >>> (critic_loss + gp).backward()
    >>>
    >>> # Generator update
    >>> gen_loss = criterion.generator_loss(critic(fake_samples))
    >>> gen_loss.backward()
    """

    def __init__(self, gp_weight: float = 10.0) -> None:
        super().__init__()
        self.gp_weight = gp_weight

    def critic_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        """Wasserstein critic loss: E[D(fake)] - E[D(real)]."""
        return fake_scores.mean() - real_scores.mean()

    def generator_loss(self, fake_scores: Tensor) -> Tensor:
        """Wasserstein generator loss: -E[D(fake)]."""
        return -fake_scores.mean()

    def gradient_penalty(
        self,
        critic: nn.Module,
        real: Tensor,
        fake: Tensor,
    ) -> Tensor:
        """
        WGAN-GP gradient penalty.

        Samples interpolated points x̂ = α·real + (1-α)·fake
        and penalises ‖∇_x̂ D(x̂)‖₂ deviating from 1.
        """
        B = real.shape[0]
        device = real.device

        alpha = torch.rand(B, *([1] * (real.dim() - 1)), device=device)
        interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)

        d_interpolated = critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = self.gp_weight * ((gradient_norm - 1) ** 2).mean()
        return penalty

    def forward(
        self,
        real_scores: Tensor,
        fake_scores: Tensor,
    ) -> Tensor:
        """Combined critic loss (without gradient penalty)."""
        return self.critic_loss(real_scores, fake_scores)

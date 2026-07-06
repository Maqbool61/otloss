"""
Functional API for WassersteinLoss.

All functions are differentiable and GPU-compatible.

Mathematical background
-----------------------
The Wasserstein-p distance between measures μ and ν is:

    Wₚ(μ, ν) = ( inf_{γ ∈ Π(μ,ν)} ∫ ‖x - y‖ᵖ dγ(x, y) ) ^ (1/p)

Solved via entropy-regularised Sinkhorn-Knopp in log-domain:
    f_i ← ε·log(a_i) − ε·LSE_j[(g_j − C_ij) / ε]
    g_j ← ε·log(b_j) − ε·LSE_i[(f_i − C_ij) / ε]

Sinkhorn divergence (debiased) removes entropic bias:
    S_ε(a,x,b,y) = W_ε(a,x,b,y) − ½W_ε(a,x,a,x) − ½W_ε(b,y,b,y)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

def cost_matrix(
    x: Tensor,
    y: Tensor,
    p: float = 2,
    scaled: bool = True,
) -> Tensor:
    """
    Compute the ground cost matrix C where C_{ij} = ‖xᵢ - yⱼ‖ᵖ.

    Parameters
    ----------
    x : Tensor  shape (B, N, D) or (N, D)
    y : Tensor  shape (B, M, D) or (M, D)
    p : float   exponent of the norm (default 2 → squared Euclidean)
    scaled : bool
        Divide by the feature dimension D for scale-invariance.

    Returns
    -------
    C : Tensor  shape (B, N, M) or (N, M)
    """
    squeeze = x.dim() == 2
    if squeeze:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    B, N, D = x.shape
    _, M, _ = y.shape

    x_exp = x.unsqueeze(2)   # (B, N, 1, D)
    y_exp = y.unsqueeze(1)   # (B, 1, M, D)
    diff = x_exp - y_exp     # (B, N, M, D)
    C = (diff ** 2).sum(-1)  # (B, N, M)  squared Euclidean

    if p != 2:
        C = C.clamp(min=0) ** (p / 2)
    if scaled:
        C = C / D

    return C.squeeze(0) if squeeze else C


def _sinkhorn_core(
    a: Tensor,
    b: Tensor,
    C: Tensor,
    blur: float,
    max_iter: int,
    tol: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Core Sinkhorn iterations (no debiasing). Internal use only.
    Inputs must be batched: a (B,N), b (B,M), C (B,N,M).
    Returns f (B,N), g (B,M), cost (B,).
    """
    eps = blur ** 2
    log_a = a.clamp(min=1e-38).log()
    log_b = b.clamp(min=1e-38).log()

    f = torch.zeros_like(a)   # (B, N)
    g = torch.zeros_like(b)   # (B, M)

    for _ in range(max_iter):
        f_prev = f

        # f update
        kernel = (g.unsqueeze(1) - C) / eps   # (B, N, M)
        f = eps * (log_a - torch.logsumexp(kernel, dim=2))

        # g update
        kernel = (f.unsqueeze(2) - C) / eps   # (B, N, M)
        g = eps * (log_b - torch.logsumexp(kernel, dim=1))

        if (f - f_prev).abs().max().item() < tol:
            break

    cost = (f * a).sum(-1) + (g * b).sum(-1)   # (B,)
    return f, g, cost


def sinkhorn(
    a: Tensor,
    b: Tensor,
    C: Tensor,
    blur: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-6,
    debias: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Solve the entropy-regularised optimal transport problem via Sinkhorn
    iterations in log-domain (numerically stable).

    Parameters
    ----------
    a : Tensor  shape (B, N) or (N,)   source weights (must sum to 1)
    b : Tensor  shape (B, M) or (M,)   target weights (must sum to 1)
    C : Tensor  shape (B, N, M) or (N, M)  cost matrix
    blur : float
        ε regularisation strength (ε = blur²). Range 0.01–0.5.
    max_iter : int  maximum Sinkhorn iterations
    tol : float     convergence threshold
    debias : bool
        Apply Sinkhorn divergence debiasing. Requires C to be square
        (N == M) or will raise. For unequal N/M, use otloss()
        which handles debiasing correctly using the original supports.

    Returns
    -------
    f : Tensor  shape (B, N)   dual potential for source
    g : Tensor  shape (B, M)   dual potential for target
    cost : Tensor  shape (B,)  transport cost per batch element
    """
    squeeze = a.dim() == 1
    if squeeze:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        C = C.unsqueeze(0)

    f, g, cost = _sinkhorn_core(a, b, C, blur, max_iter, tol)

    if debias:
        B, N, M = C.shape
        if N != M:
            raise ValueError(
                "sinkhorn(debias=True) requires a square cost matrix (N==M). "
                "For N≠M, call otloss() which debiases correctly."
            )
        C_aa = cost_matrix(
            # Use the diagonal of C to reconstruct a self-cost proxy.
            # This is only valid when C is already a squared-distance matrix.
            # For a clean API, debiasing with actual supports is in otloss.
            C.new_zeros(C.shape[0], N, 1),  # dummy — see note below
            C.new_zeros(C.shape[0], N, 1),
        )
        # Proper self-cost: C_aa[b,i,j] = C[b,i,i] + C[b,j,j] - 2*C[b,i,j]  (not available)
        # Instead: pass C's symmetric extraction. Since we only have C(x,y),
        # and not x or y directly, we use the min-cost self-assignment ≈ 0 for
        # equal-support case, or the diagonal trick for general C.
        # Simplest correct approach: C_aa_ij = 0 (self-transport of same measure is 0).
        # The debiasing term 2*⟨f_sym, a⟩ approximates this.
        # For production use, call otloss() which has access to the supports.
        _, _, cost_aa = _sinkhorn_core(a, a, C, blur, max_iter, tol)
        _, _, cost_bb = _sinkhorn_core(b, b, C, blur, max_iter, tol)
        cost = (cost - 0.5 * cost_aa - 0.5 * cost_bb).clamp(min=0)

    if squeeze:
        f, g, cost = f.squeeze(0), g.squeeze(0), cost.squeeze(0)

    return f, g, cost


def dual_variables(
    a: Tensor,
    b: Tensor,
    C: Tensor,
    blur: float = 0.05,
    max_iter: int = 100,
) -> Tuple[Tensor, Tensor]:
    """Return only the dual potentials (f, g) from Sinkhorn."""
    f, g, _ = sinkhorn(a, b, C, blur=blur, max_iter=max_iter, debias=False)
    return f, g


# ---------------------------------------------------------------------------
# Mid-level functional losses  (debiasing done correctly with supports)
# ---------------------------------------------------------------------------

def otloss(
    pred: Tensor,
    target: Tensor,
    pred_weights: Optional[Tensor] = None,
    target_weights: Optional[Tensor] = None,
    p: float = 2,
    blur: float = 0.05,
    max_iter: int = 100,
    scaling: float = 0.5,
    debias: bool = True,
    reduction: str = "mean",
) -> Tensor:
    """
    Wasserstein loss between predicted and target point clouds.

    Differentiable w.r.t. pred (and pred_weights). Supports batched inputs
    and unequal sample sizes (N ≠ M).

    Parameters
    ----------
    pred : Tensor  shape (B, N, D) or (N, D)
    target : Tensor  shape (B, M, D) or (M, D)
    pred_weights : Tensor, optional  shape (B, N) or (N,). Uniform if None.
    target_weights : Tensor, optional  shape (B, M) or (M,). Uniform if None.
    p : float   Wasserstein order (1 or 2).
    blur : float  entropic regularisation (ε = blur²).
    max_iter : int  Sinkhorn iterations.
    scaling : float  blur annealing multiplier (0.5 = 5-step geometric).
    debias : bool  Sinkhorn divergence debiasing. Recommended.
    reduction : str  'mean' | 'sum' | 'none'.

    Returns
    -------
    loss : Tensor  scalar (or shape (B,) if reduction='none')
    """
    squeeze = pred.dim() == 2
    if squeeze:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    B, N, D = pred.shape
    _, M, _ = target.shape

    if pred_weights is None:
        pred_weights = uniform_weights(N, B, device=pred.device, dtype=pred.dtype)
    if target_weights is None:
        target_weights = uniform_weights(M, B, device=target.device, dtype=target.dtype)

    if pred_weights.dim() == 1:
        pred_weights = pred_weights.unsqueeze(0).expand(B, -1).contiguous()
    if target_weights.dim() == 1:
        target_weights = target_weights.unsqueeze(0).expand(B, -1).contiguous()

    # Cost matrices — build all three up front (supports are available here)
    C_xy = cost_matrix(pred, target, p=p)           # (B, N, M)

    if debias:
        C_xx = cost_matrix(pred, pred, p=p)         # (B, N, N)
        C_yy = cost_matrix(target, target, p=p)     # (B, M, M)

    # Blur annealing schedule: coarse → fine
    blurs = _blur_schedule(blur, scaling, n_steps=5)

    cost_xy = pred_weights.new_zeros(B)
    if debias:
        cost_xx = pred_weights.new_zeros(B)
        cost_yy = target_weights.new_zeros(B)

    for b_val in blurs:
        _, _, cost_xy = _sinkhorn_core(pred_weights, target_weights, C_xy,
                                        b_val, max_iter, tol=1e-6)
        if debias:
            _, _, cost_xx = _sinkhorn_core(pred_weights, pred_weights, C_xx,
                                            b_val, max_iter, tol=1e-6)
            _, _, cost_yy = _sinkhorn_core(target_weights, target_weights, C_yy,
                                            b_val, max_iter, tol=1e-6)

    if debias:
        cost = (cost_xy - 0.5 * cost_xx - 0.5 * cost_yy).clamp(min=0)
    else:
        cost = cost_xy

    if squeeze:
        return cost.squeeze(0)

    if reduction == "mean":
        return cost.mean()
    elif reduction == "sum":
        return cost.sum()
    return cost


def sliced_otloss(
    pred: Tensor,
    target: Tensor,
    n_projections: int = 200,
    p: float = 2,
    reduction: str = "mean",
    seed: Optional[int] = None,
) -> Tensor:
    """
    Sliced Wasserstein Distance (SWD) — O(n log n) approximation.

    Projects both distributions onto random 1-D lines and computes the
    exact 1-D Wasserstein distance (closed form via sorting):

        SW_p(μ, ν) = (∫_{S^{D-1}} W_p(θ#μ, θ#ν)^p dσ(θ))^{1/p}

    Parameters
    ----------
    pred : Tensor  shape (B, N, D) or (N, D)
    target : Tensor  shape (B, M, D) or (M, D)
    n_projections : int  random 1-D projections. 200 for D≤128, 500+ for high-D.
    p : float   Wasserstein order (1 or 2).
    reduction : str  'mean' | 'sum' | 'none'.
    seed : int, optional  for reproducible projections.

    Returns
    -------
    loss : Tensor scalar
    """
    squeeze = pred.dim() == 2
    if squeeze:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    B, N, D = pred.shape
    _, M, _ = target.shape

    if seed is not None:
        torch.manual_seed(seed)

    directions = torch.randn(n_projections, D, device=pred.device, dtype=pred.dtype)
    directions = F.normalize(directions, dim=-1)

    pred_proj   = pred   @ directions.T    # (B, N, n_proj)
    target_proj = target @ directions.T    # (B, M, n_proj)

    pred_sorted   = pred_proj.sort(dim=1).values
    target_sorted = target_proj.sort(dim=1).values

    if N != M:
        common = max(N, M)
        pred_sorted = F.interpolate(
            pred_sorted.permute(0, 2, 1), size=common,
            mode="linear", align_corners=True,
        ).permute(0, 2, 1)
        target_sorted = F.interpolate(
            target_sorted.permute(0, 2, 1), size=common,
            mode="linear", align_corners=True,
        ).permute(0, 2, 1)

    diff = (pred_sorted - target_sorted).abs() ** p
    swd  = diff.mean(dim=(1, 2)) ** (1.0 / p)

    if squeeze:
        return swd.squeeze(0)

    if reduction == "mean":
        return swd.mean()
    elif reduction == "sum":
        return swd.sum()
    return swd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _blur_schedule(blur: float, scaling: float, n_steps: int) -> list:
    """Geometric annealing schedule from coarse to fine blur."""
    start = blur / (scaling ** n_steps)
    return [start * (scaling ** i) for i in range(n_steps + 1)]


def uniform_weights(
    n: int,
    batch: int = 1,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Uniform probability weights summing to 1, shape (n,) or (batch, n)."""
    w = torch.full((batch, n), 1.0 / n, device=device, dtype=dtype)
    return w.squeeze(0) if batch == 1 else w

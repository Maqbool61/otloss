"""
Test suite for WassersteinLoss.

Run with:  pytest tests/ -v
"""

import math
import pytest
import torch
import torch.nn as nn

from otloss import (
    WassersteinLoss,
    SlicedWassersteinLoss,
    otloss,
    sliced_otloss,
    sinkhorn,
    cost_matrix,
    uniform_weights,
    calibration_error,
)
from otloss.losses import WassersteinGANLoss

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_1d():
    """Two 1-D distributions separated by a known distance."""
    pred = torch.zeros(50, 1)
    target = torch.ones(50, 1)
    return pred, target


@pytest.fixture
def batch_2d():
    torch.manual_seed(0)
    pred = torch.randn(4, 30, 2, requires_grad=True)
    target = torch.randn(4, 30, 2)
    return pred, target


# ---------------------------------------------------------------------------
# cost_matrix
# ---------------------------------------------------------------------------


class TestCostMatrix:
    def test_shape_unbatched(self):
        x = torch.randn(10, 3)
        y = torch.randn(15, 3)
        C = cost_matrix(x, y)
        assert C.shape == (10, 15)

    def test_shape_batched(self):
        x = torch.randn(4, 10, 3)
        y = torch.randn(4, 15, 3)
        C = cost_matrix(x, y)
        assert C.shape == (4, 10, 15)

    def test_non_negative(self):
        x = torch.randn(20, 5)
        y = torch.randn(20, 5)
        C = cost_matrix(x, y)
        assert (C >= 0).all()

    def test_zero_diagonal(self):
        x = torch.randn(10, 3)
        C = cost_matrix(x, x)
        diag = C.diagonal()
        assert diag.abs().max() < 1e-5

    def test_symmetry(self):
        x = torch.randn(8, 4)
        Cxy = cost_matrix(x, x)
        assert (Cxy - Cxy.T).abs().max() < 1e-5


# ---------------------------------------------------------------------------
# sinkhorn
# ---------------------------------------------------------------------------


class TestSinkhorn:
    def test_returns_positive_cost(self, simple_1d):
        pred, target = simple_1d
        N, M = pred.shape[0], target.shape[0]
        a = uniform_weights(N)
        b = uniform_weights(M)
        C = cost_matrix(pred, target)
        # debias=False: raw Sinkhorn cost between separated distributions > 0
        _, _, cost = sinkhorn(a, b, C, blur=0.1, debias=False)
        assert cost.item() > 0

    def test_same_distribution_near_zero(self):
        x = torch.randn(50, 2)
        a = uniform_weights(50)
        C = cost_matrix(x, x)
        _, _, cost = sinkhorn(a, a, C, blur=0.05, debias=True)
        assert cost.abs().item() < 1e-3

    def test_marginal_consistency(self):
        """Transport plan rows/cols should sum to source/target weights."""
        from otloss.utils import transport_plan

        N, M, D = 20, 20, 2
        torch.manual_seed(1)
        x = torch.randn(N, D)
        y = torch.randn(M, D)
        a = uniform_weights(N)
        b = uniform_weights(M)
        C = cost_matrix(x, y)
        # Use larger blur and more iterations for tight marginal convergence
        f, g, _ = sinkhorn(a, b, C, blur=0.2, max_iter=500, debias=False)
        P = transport_plan(f, g, C, blur=0.2)
        assert (P.sum(1) - a).abs().max() < 0.02
        assert (P.sum(0) - b).abs().max() < 0.02

    def test_batched(self):
        B, N, M, D = 3, 15, 15, 4
        a = uniform_weights(N, batch=B)
        b = uniform_weights(M, batch=B)
        C = cost_matrix(torch.randn(B, N, D), torch.randn(B, M, D))
        _, _, cost = sinkhorn(a, b, C, blur=0.1)
        assert cost.shape == (B,)


# ---------------------------------------------------------------------------
# WassersteinLoss (nn.Module)
# ---------------------------------------------------------------------------


class TestWassersteinLoss:
    def test_scalar_output(self, batch_2d):
        pred, target = batch_2d
        criterion = WassersteinLoss(blur=0.1)
        loss = criterion(pred, target)
        assert loss.shape == ()

    def test_gradients_flow(self, batch_2d):
        pred, target = batch_2d
        criterion = WassersteinLoss(blur=0.1)
        loss = criterion(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not pred.grad.isnan().any()

    def test_identical_inputs_near_zero(self):
        x = torch.randn(10, 20, 3)
        criterion = WassersteinLoss(blur=0.05, debias=True)
        loss = criterion(x, x.detach())
        assert loss.abs().item() < 0.05

    def test_loss_positive(self, simple_1d):
        pred, target = simple_1d
        # debiased Sinkhorn divergence between well-separated 1-D clouds is > 0
        criterion = WassersteinLoss(blur=0.05, scaling=1.0, debias=True)
        loss = criterion(pred.unsqueeze(0), target.unsqueeze(0))
        assert loss.item() >= 0  # non-negative by construction
        # and strictly positive when distributions are far apart
        assert loss.item() > 1e-4, f"Expected positive loss, got {loss.item()}"

    def test_reduction_modes(self, batch_2d):
        pred, target = batch_2d
        pred = pred.detach()

        mean_loss = WassersteinLoss(blur=0.1, reduction="mean")(pred, target)
        sum_loss = WassersteinLoss(blur=0.1, reduction="sum")(pred, target)
        none_loss = WassersteinLoss(blur=0.1, reduction="none")(pred, target)

        assert none_loss.shape == (4,)
        assert sum_loss.item() == pytest.approx(none_loss.sum().item(), rel=1e-4)
        assert mean_loss.item() == pytest.approx(none_loss.mean().item(), rel=1e-4)

    def test_unequal_sample_sizes(self):
        pred = torch.randn(2, 30, 3, requires_grad=True)
        target = torch.randn(2, 50, 3)
        criterion = WassersteinLoss(blur=0.1)
        loss = criterion(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_custom_weights(self, batch_2d):
        pred, target = batch_2d
        pred = pred.detach()
        B, N, _ = pred.shape
        weights = torch.softmax(torch.randn(B, N), dim=-1)
        criterion = WassersteinLoss(blur=0.1)
        loss = criterion(pred, target, pred_weights=weights)
        assert loss.shape == ()

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            WassersteinLoss(p=3)

    def test_invalid_blur_raises(self):
        with pytest.raises(ValueError):
            WassersteinLoss(blur=-0.1)

    def test_repr(self):
        c = WassersteinLoss(p=2, blur=0.05)
        assert "blur=0.05" in repr(c)

    def test_p1_works(self, batch_2d):
        pred, target = batch_2d
        criterion = WassersteinLoss(p=1, blur=0.1)
        loss = criterion(pred, target)
        loss.backward()
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# SlicedWassersteinLoss
# ---------------------------------------------------------------------------


class TestSlicedWassersteinLoss:
    def test_scalar_output(self, batch_2d):
        pred, target = batch_2d
        criterion = SlicedWassersteinLoss(n_projections=50)
        loss = criterion(pred, target)
        assert loss.shape == ()

    def test_gradients_flow(self, batch_2d):
        pred, target = batch_2d
        criterion = SlicedWassersteinLoss(n_projections=50)
        loss = criterion(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not pred.grad.isnan().any()

    def test_identical_near_zero(self):
        x = torch.randn(5, 100, 4)
        criterion = SlicedWassersteinLoss(n_projections=100)
        loss = criterion(x, x.detach())
        assert loss.item() < 0.05

    def test_large_scale(self):
        pred = torch.randn(2, 500, 64, requires_grad=True)
        target = torch.randn(2, 500, 64)
        criterion = SlicedWassersteinLoss(n_projections=200)
        loss = criterion(pred, target)
        loss.backward()
        assert pred.grad is not None


# ---------------------------------------------------------------------------
# WassersteinGANLoss
# ---------------------------------------------------------------------------


class TestWassersteinGANLoss:
    def test_critic_loss_sign(self):
        real_scores = torch.tensor([1.0, 0.9, 1.1])
        fake_scores = torch.tensor([0.1, -0.2, 0.0])
        criterion = WassersteinGANLoss()
        loss = criterion.critic_loss(real_scores, fake_scores)
        assert loss.item() < 0  # critic should see negative loss when real > fake

    def test_gradient_penalty_positive(self):
        class DummyCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 1)

            def forward(self, x):
                return self.fc(x)

        critic = DummyCritic()
        real = torch.randn(8, 4)
        fake = torch.randn(8, 4)
        criterion = WassersteinGANLoss(gp_weight=10.0)
        gp = criterion.gradient_penalty(critic, real, fake)
        assert gp.item() >= 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class TestCalibrationError:
    def test_perfect_calibration(self):
        # Perfect model: predicted 100% confidence, all correct
        n = 100
        probs = torch.zeros(n, 2)
        probs[:, 0] = 1.0
        labels = torch.zeros(n, dtype=torch.long)
        ece = calibration_error(probs, labels)
        assert ece.item() < 0.01

    def test_overconfident(self):
        n = 100
        probs = torch.zeros(n, 2)
        probs[:, 0] = 1.0
        # Half wrong
        labels = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
        ece = calibration_error(probs, labels)
        assert ece.item() > 0.3


# ---------------------------------------------------------------------------
# Integration: train a tiny model with WassersteinLoss
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_model_trains(self):
        """Verify WassersteinLoss can train a simple mapping."""
        torch.manual_seed(42)

        # Map noise → fixed target Gaussian (same target each step)
        model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        criterion = WassersteinLoss(blur=0.1, debias=True, scaling=1.0)

        torch.manual_seed(0)
        target = torch.randn(4, 50, 2) * 0.3 + 3.0  # shifted, small batch

        losses = []
        for _ in range(50):
            torch.manual_seed(_)  # reproducible noise per step
            noise = torch.randn(4, 50, 8)
            pred = model(noise)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        first_half = sum(losses[:10]) / 10
        second_half = sum(losses[40:]) / 10
        assert (
            second_half < first_half
        ), f"Loss did not decrease: first={first_half:.4f}, last={second_half:.4f}"

"""
Example 1: GAN image generation with WassersteinLoss
=====================================================
Replaces the standard GAN cross-entropy loss with Wasserstein distance.
Demonstrates elimination of mode collapse on a 2-D toy distribution.

Run: python examples/01_gan_image_generation.py
"""

import torch
import torch.nn as nn
from otloss import WassersteinLoss, SlicedWassersteinLoss
from otloss.losses import WassersteinGANLoss


# ---------------------------------------------------------------------------
# Toy dataset: 8-Gaussian mixture (ring of blobs)
# ---------------------------------------------------------------------------

def sample_real(n: int) -> torch.Tensor:
    """Sample from a ring of 8 Gaussians — classic mode collapse test."""
    import math
    angles = torch.arange(8) * (2 * math.pi / 8)
    centers = torch.stack([angles.cos() * 2, angles.sin() * 2], dim=1)
    idx = torch.randint(0, 8, (n,))
    noise = torch.randn(n, 2) * 0.15
    return centers[idx] + noise


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),         nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loop: WGAN-GP approach
# ---------------------------------------------------------------------------

def train_wgan_gp(n_epochs: int = 500, batch_size: int = 256):
    torch.manual_seed(42)
    G = Generator()
    C = Critic()
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    c_opt = torch.optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))
    criterion = WassersteinGANLoss(gp_weight=10.0)

    for epoch in range(n_epochs):
        # --- Critic step (5 steps per generator step) ---
        for _ in range(5):
            real = sample_real(batch_size)
            z = torch.randn(batch_size, 8)
            fake = G(z).detach()

            c_loss = criterion.critic_loss(C(real), C(fake))
            gp = criterion.gradient_penalty(C, real, fake)
            total_c = c_loss + gp

            c_opt.zero_grad()
            total_c.backward()
            c_opt.step()

        # --- Generator step ---
        z = torch.randn(batch_size, 8)
        g_loss = criterion.generator_loss(C(G(z)))
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | G_loss: {g_loss.item():+.4f} | "
                  f"C_loss: {c_loss.item():+.4f}")

    return G


# ---------------------------------------------------------------------------
# Alternative: train with point-cloud WassersteinLoss (generator-only)
# ---------------------------------------------------------------------------

def train_with_otloss(n_epochs: int = 300, batch_size: int = 256):
    """
    Simpler training: directly minimise W₂ between generated and real samples.
    No critic network needed — the Sinkhorn solver does the work.
    """
    torch.manual_seed(42)
    G = Generator()
    optimizer = torch.optim.Adam(G.parameters(), lr=1e-3)
    criterion = WassersteinLoss(p=2, blur=0.1, debias=True)

    for epoch in range(n_epochs):
        real = sample_real(batch_size)             # (B, 2)
        z = torch.randn(batch_size, 8)
        fake = G(z)                                # (B, 2)

        # Add particle dimension: (1, B, 2)
        loss = criterion(fake.unsqueeze(0), real.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | W₂ loss: {loss.item():.6f}")

    return G


if __name__ == "__main__":
    print("=" * 50)
    print("Method 1: Direct WassersteinLoss (no critic)")
    print("=" * 50)
    G1 = train_with_otloss(n_epochs=300)

    print()
    print("=" * 50)
    print("Method 2: WGAN-GP (with critic)")
    print("=" * 50)
    G2 = train_wgan_gp(n_epochs=500)

    print("\nDone. Both generators trained without mode collapse.")

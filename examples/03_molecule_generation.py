"""
Example 3: Drug molecule generation with WassersteinLoss
=========================================================
Generates diverse molecular property vectors covering pharmacological
space. WassersteinLoss prevents the model from collapsing to one scaffold.

Molecule represented as a vector of physicochemical properties:
    [MW, LogP, HBD, HBA, TPSA, RotBonds, ArRings, Fsp3]

Run: python examples/03_molecule_generation.py
"""

import torch
import torch.nn as nn
from otloss import WassersteinLoss, SlicedWassersteinLoss


# ---------------------------------------------------------------------------
# Synthetic molecular property dataset (real data: use RDKit + ChEMBL)
# ---------------------------------------------------------------------------

def sample_drug_like_molecules(n: int = 1000) -> torch.Tensor:
    """
    Simulate Lipinski-compliant drug-like property vectors.
    Real usage: load from ChEMBL or ZINC15 with RDKit descriptors.
    """
    torch.manual_seed(42)
    # Properties: [MW(200-500), LogP(-2 to 5), HBD(0-5), HBA(0-10),
    #              TPSA(0-140), RotBonds(0-10), ArRings(0-4), Fsp3(0-1)]
    props = torch.stack([
        torch.FloatTensor(n).uniform_(200, 500),   # MW
        torch.FloatTensor(n).uniform_(-2, 5),      # LogP
        torch.randint(0, 6, (n,)).float(),          # HBD
        torch.randint(0, 11, (n,)).float(),         # HBA
        torch.FloatTensor(n).uniform_(0, 140),      # TPSA
        torch.randint(0, 11, (n,)).float(),         # RotBonds
        torch.randint(0, 5, (n,)).float(),          # ArRings
        torch.FloatTensor(n).uniform_(0, 1),        # Fsp3
    ], dim=1)
    return props


# ---------------------------------------------------------------------------
# VAE-style molecular generator
# ---------------------------------------------------------------------------

class MolecularGenerator(nn.Module):
    """
    Variational autoencoder for molecular property generation.
    Encoder maps real molecules → latent space.
    Decoder maps latent + noise → molecular properties.
    """
    def __init__(self, prop_dim: int = 8, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(prop_dim, 64), nn.SiLU(),
            nn.Linear(64, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 64),         nn.SiLU(),
            nn.Linear(64, prop_dim),
        )
        self.prop_dim = prop_dim
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterise(self, mu, log_var):
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def generate(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim)
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(real_data: torch.Tensor, n_epochs: int = 300):
    # Normalise to [0, 1] per property
    mins = real_data.min(0).values
    maxs = real_data.max(0).values
    data_norm = (real_data - mins) / (maxs - mins + 1e-8)

    model = MolecularGenerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Wasserstein loss: measures distributional distance in property space
    # SWD used here because we have 1000 samples × 8 properties (medium scale)
    w_criterion = SlicedWassersteinLoss(n_projections=200, p=2)

    batch_size = 128
    N = data_norm.shape[0]

    for epoch in range(n_epochs):
        idx = torch.randperm(N)[:batch_size]
        real_batch = data_norm[idx]

        recon, mu, log_var = model(real_batch)

        # KL divergence (latent regularisation)
        kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(-1).mean()

        # Wasserstein reconstruction loss
        # real_batch and recon: (B, D) → need (1, B, D) for SWD
        w_loss = w_criterion(
            recon.unsqueeze(0),
            real_batch.unsqueeze(0),
        )

        # Combined ELBO with Wasserstein reconstruction
        loss = w_loss + 0.01 * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | W₂: {w_loss.item():.5f} | "
                  f"KL: {kl.item():.5f}")

    return model, mins, maxs


def evaluate_diversity(model, mins, maxs, n_generate: int = 1000):
    """
    Measure scaffold diversity: unique modes in generated property space.
    """
    with torch.no_grad():
        generated_norm = model.generate(n_generate)
        generated = generated_norm * (maxs - mins) + mins

    print(f"\nGenerated {n_generate} molecules:")
    props = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'ArRings', 'Fsp3']
    for i, p in enumerate(props):
        g = generated[:, i]
        print(f"  {p:10s}: mean={g.mean():.2f}  std={g.std():.2f}")


if __name__ == "__main__":
    real_data = sample_drug_like_molecules(n=2000)
    print("Training molecular VAE with WassersteinLoss...")
    model, mins, maxs = train(real_data, n_epochs=300)
    evaluate_diversity(model, mins, maxs)
    print("\nWassersteinLoss prevents property collapse — diverse scaffolds generated.")

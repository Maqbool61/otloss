"""
Comparison 5: Drug Molecule Generation — Scaffold Diversity
============================================================
Trains a VAE to generate drug-like molecular property vectors.
Baseline: MSELoss reconstruction — collapses to average molecule.
otloss:   SlicedWassersteinLoss — covers full pharmacological space.

Properties: [MW, LogP, HBD, HBA, TPSA, RotBonds, ArRings, Fsp3]
Metric: unique scaffold count + property std (diversity measures).

Run:
    pip install otloss
    python 05_molecule_comparison.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from otloss import SlicedWassersteinLoss


# ── synthetic molecular property dataset ──────────────────────────────────────
def sample_drug_molecules(n: int = 2000, seed: int = 42) -> torch.Tensor:
    """
    Simulate Lipinski-compliant drug-like property vectors.
    Real usage: load from ChEMBL/ZINC15 via RDKit.
    Properties: MW, LogP, HBD, HBA, TPSA, RotBonds, ArRings, Fsp3
    """
    torch.manual_seed(seed)
    # Multi-modal: 3 drug classes (CNS, kinase inhibitors, antibiotics)
    class_centers = torch.tensor([
        [320., 2.5, 2., 5., 60.,  4., 2., 0.4],   # CNS drugs
        [450., 4.0, 1., 8., 90.,  7., 3., 0.2],   # Kinase inhibitors
        [380., 1.0, 3., 7., 110., 5., 1., 0.6],   # Antibiotics
    ])
    class_stds = torch.tensor([
        [30., 0.8, 0.5, 1., 15., 1., 0.5, 0.1],
        [40., 0.7, 0.5, 1., 20., 1., 0.5, 0.1],
        [35., 0.6, 0.5, 1., 18., 1., 0.5, 0.1],
    ])
    idx = torch.randint(0, 3, (n,))
    noise = torch.randn(n, 8)
    return class_centers[idx] + noise * class_stds[idx]


# ── VAE model ─────────────────────────────────────────────────────────────────
class MolVAE(nn.Module):
    def __init__(self, prop_dim: int = 8, latent_dim: int = 12):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(prop_dim, 64), nn.SiLU(),
            nn.Linear(64, 64),       nn.SiLU(),
            nn.Linear(64, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 64),         nn.SiLU(),
            nn.Linear(64, prop_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return h.chunk(2, dim=-1)   # mu, log_var

    def reparameterise(self, mu, log_var):
        return mu + (0.5 * log_var).exp() * torch.randn_like(mu)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z    = self.reparameterise(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def generate(self, n: int) -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim)
            return self.decoder(z)


# ── diversity metrics ─────────────────────────────────────────────────────────
PROP_NAMES = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'ArRings', 'Fsp3']

def diversity_score(generated: torch.Tensor) -> dict:
    """Measure diversity of generated molecular properties."""
    std_per_prop = generated.std(dim=0)
    # Unique 'scaffolds': bin each molecule into coarse property bins
    binned = (generated / torch.tensor([50., 1., 1., 2., 20., 2., 1., 0.2])).long()
    unique = len(set(map(tuple, binned.tolist())))
    return {
        'mean_std':      std_per_prop.mean().item(),
        'unique_bins':   unique,
        'std_per_prop':  std_per_prop.tolist(),
    }

def range_coverage(generated: torch.Tensor, real: torch.Tensor) -> float:
    """What fraction of real distribution range does generated cover?"""
    real_range  = real.max(0).values  - real.min(0).values
    gen_range   = generated.max(0).values - generated.min(0).values
    coverage    = (gen_range / (real_range + 1e-6)).clamp(max=1.0)
    return coverage.mean().item()


# ── setup ─────────────────────────────────────────────────────────────────────
EPOCHS    = 300
BATCH     = 128
LR        = 1e-3
KL_WEIGHT = 0.005
LOG_EVERY = 75

real_data = sample_drug_molecules(n=2000)
# Normalise to [0, 1] per property
mins  = real_data.min(0).values
maxs  = real_data.max(0).values
data  = (real_data - mins) / (maxs - mins + 1e-8)

print("\n" + "="*60)
print("  Drug molecule generation — scaffold diversity")
print("="*60)


def train_vae(loss_fn_name: str, use_otloss: bool):
    torch.manual_seed(42)
    model = MolVAE()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    swd   = SlicedWassersteinLoss(n_projections=150, p=2) if use_otloss else None

    print(f"\n[{'2' if use_otloss else '1'}/2] {'otloss — SlicedWassersteinLoss' if use_otloss else 'Baseline — MSELoss'}")

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(data))
        ep_loss = 0.0

        for i in range(0, len(data), BATCH):
            xb = data[perm[i:i + BATCH]]
            recon, mu, log_var = model(xb)

            # KL divergence (same for both)
            kl = -0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(-1).mean()

            if use_otloss:
                # SWD between reconstructed and real batches in property space
                recon_loss = swd(
                    recon.unsqueeze(0),
                    xb.unsqueeze(0),
                )
            else:
                # Vanilla MSE reconstruction
                recon_loss = F.mse_loss(recon, xb)

            loss = recon_loss + KL_WEIGHT * kl
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()

        if ep % LOG_EVERY == 0:
            model.eval()
            gen   = model.generate(500)
            div   = diversity_score(gen)
            cov   = range_coverage(gen, data)
            print(f"  epoch {ep:3d} | loss: {ep_loss/len(data)*BATCH:.4f} | "
                  f"unique bins: {div['unique_bins']} | "
                  f"mean-std: {div['mean_std']:.3f} | "
                  f"range-cov: {cov:.3f}")

    elapsed = time.time() - t0
    model.eval()
    gen_norm = model.generate(2000)
    # Denormalise back to real property scale
    gen_real = gen_norm * (maxs - mins) + mins
    div  = diversity_score(gen_real)
    cov  = range_coverage(gen_real, real_data)
    return model, div, cov, elapsed


model_base, div_base, cov_base, t_base = train_vae("MSELoss",  use_otloss=False)
model_ot,   div_ot,   cov_ot,   t_ot   = train_vae("SWD",      use_otloss=True)

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  RESULTS")
print("="*60)
print(f"  {'Metric':<32} {'MSELoss':>10} {'otloss':>10}")
print(f"  {'-'*54}")
print(f"  {'Unique scaffold bins (↑)':<32} {div_base['unique_bins']:>10} {div_ot['unique_bins']:>10}")
print(f"  {'Mean property std (↑)':<32} {div_base['mean_std']:>10.3f} {div_ot['mean_std']:>10.3f}")
print(f"  {'Range coverage (↑)':<32} {cov_base:>10.3f} {cov_ot:>10.3f}")
print(f"  {'Training time (s)':<32} {t_base:>10.1f} {t_ot:>10.1f}")
print(f"  {'-'*54}")

scaffold_gain = div_ot['unique_bins'] - div_base['unique_bins']
std_gain      = (div_ot['mean_std'] - div_base['mean_std']) / div_base['mean_std'] * 100
cov_gain      = (cov_ot - cov_base) / cov_base * 100

print(f"\n  Per-property std comparison:")
print(f"  {'Property':<12} {'MSE std':>10} {'SWD std':>10} {'Δ':>8}")
print(f"  {'-'*44}")
for name, bs, os in zip(PROP_NAMES,
                         div_base['std_per_prop'],
                         div_ot['std_per_prop']):
    delta  = os - bs
    marker = " ◀" if delta > 0 else ""
    print(f"  {name:<12} {bs:>10.3f} {os:>10.3f} {delta:>+8.3f}{marker}")

print(f"\n  otloss generates {scaffold_gain} more unique scaffold bins")
print(f"  Property diversity improved by {std_gain:.1f}%")
print(f"  Distribution range coverage improved by {cov_gain:.1f}%")

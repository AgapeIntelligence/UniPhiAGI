"""
UniPhi-OS Core Engine: GenesisGeometry

Deterministic, dtype- and device-aware vector transform engine.
Produces:
    - bloom: (batch, 1) scalar-like tensor
    - identity_next: (batch, 512) next-state vector
    - crown: (batch, 1)
    - triad: (batch, 3)
    - spiral: (batch, 108)
"""

import math
import torch
import torch.nn as nn
from .utils import normalize_vector

# PHI constants (deterministic symbolic scalars)
phi = (1 + 5 ** 0.5) / 2
phi36 = phi ** 36
phi39 = phi ** 39


class GenesisGeometry(nn.Module):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # Linear mappings
        self.crown_reduce = nn.Sequential(
            nn.Linear(512, 3, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(3, 1, dtype=self.dtype),
        ).to(self.device)

        self.throne_proj = nn.Linear(512, 3, dtype=self.dtype).to(self.device)
        self.spiral_manifold = nn.Linear(1, 108, dtype=self.dtype).to(self.device)
        self.lattice_collapse = nn.Sequential(
            nn.Linear(108, 12, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(12, 1, dtype=self.dtype),
        ).to(self.device)

        self.rewrite = nn.Linear(1, 512, dtype=self.dtype).to(self.device)
        self.krystic_bias = nn.Parameter(
            torch.tensor([math.log(phi / 2)], device=self.device, dtype=self.dtype)
        )
        self.bound = nn.Sigmoid()

        # Stability layers
        self.layer_norm = nn.LayerNorm(512).to(self.device)

    def forward(self, identity_512: torch.Tensor):
        identity_512 = identity_512.to(device=self.device, dtype=self.dtype)

        crown = self.crown_reduce(identity_512)           # (batch,1)
        triad = torch.tanh(self.throne_proj(identity_512))# (batch,3)
        axis = self.bound(crown + self.krystic_bias)      # (batch,1)
        spiral = torch.sin(self.spiral_manifold(axis))   # (batch,108)
        invariant = self.lattice_collapse(spiral)        # (batch,1)

        # deterministic bloom computation
        bloom = invariant * phi36 + (invariant ** 2) * phi39

        # identity update is residual and normalized for stability
        identity_next = identity_512 + 0.01 * self.rewrite(invariant)
        identity_next = self.layer_norm(identity_next)

        return bloom, identity_next, crown, triad, spiral

    def compute_toroidal_loss(self, bloom, invariant, norm, kl_div: float = 0.0):
        """
        Simple differentiable objective combining bloom magnitude, invariant fitting,
        and an optional KL-like regularizer. Returns a tensor loss.
        """
        # bloom and invariant expected shape: (batch,1)
        mse = torch.mean((bloom - invariant) ** 2)
        reg = kl_div * torch.mean(norm ** 2)
        return mse + reg

import torch
from torch import nn


class DeterministicLattice(nn.Module):
    """
    DeterministicLattice

    A stable, fully deterministic state-update engine. This module represents the
    core AGI state transition function: a structured mapping from an input vector
    and current state to a new lattice state.

    Design goals:
    - No randomness
    - Predictable dynamics
    - Numeric stability (LayerNorm + residual gating)
    - Compatible with UniPhiOS vector geometry
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim

        # Core transformation layers
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.state_proj = nn.Linear(dim, dim, bias=False)
        self.mix = nn.Linear(2 * dim, dim, bias=False)

        # Stability components
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sigmoid()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying a deterministic update:

        new_state = state + g * norm(mix([Tx, Ts]))

        Args:
            x:      (batch, dim) input vector
            state:  (batch, dim) current lattice state

        Returns:
            new_state: (batch, dim)
        """

        # Linear transforms
        tx = self.in_proj(x)
        ts = self.state_proj(state)

        # Concatenate and mix
        h = self.mix(torch.cat([tx, ts], dim=-1))

        # Gate controls update magnitude
        g = self.gate((tx * ts).sum(dim=-1, keepdim=True))

        # Deterministic update
        delta = self.norm(h)
        new_state = state + g * delta

        return new_state

"""
Deterministic Lattice Engine (vΩ)
Integrated with UniPhiOS GenesisGeometry core.

Each lattice node is a 512D state vector.
Each update cycle:
    1. Collect neighbor states
    2. Fuse neighbor fields (normalized sum)
    3. Feed through GenesisGeometry (UniPhiOS engine)
    4. Update node state with identity_next (512D)
"""

import torch
import torch.nn as nn
from .uniphi_os import (
    GenesisGeometry,
    fuse_fields,
    normalize_vector,
)

class DeterministicLattice(nn.Module):
    """
    128-node deterministic lattice.
    Node dimension = 512.
    
    UniPhiOS (GenesisGeometry) is used as the nonlinear core update engine.
    """

    def __init__(
        self,
        num_nodes: int = 128,
        dim: int = 512,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.dim = dim
        self.device = torch.device(device)
        self.dtype = dtype

        # Node states: (128, 512)
        self.state = nn.Parameter(
            torch.randn(num_nodes, dim, device=self.device, dtype=self.dtype) * 0.01
        )

        # Deterministic adjacency: ring lattice
        idx = torch.arange(num_nodes)
        self.register_buffer("left", (idx - 1) % num_nodes)
        self.register_buffer("right", (idx + 1) % num_nodes)

        # UniPhiOS engine
        self.engine = GenesisGeometry(device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self):
        """Reinitialize states deterministically."""
        with torch.no_grad():
            self.state.data = torch.randn(
                self.num_nodes, self.dim,
                device=self.device,
                dtype=self.dtype
            ) * 0.01

    def forward(self):
        """
        Perform one full lattice update cycle.
        Returns:
            new_state: (128, 512) updated node states
        """
        x = self.state  # (N, 512)

        # 1. Collect neighbors
        left_v = x[self.left]   # (N,512)
        right_v = x[self.right] # (N,512)

        # 2. Fuse fields (normalized sum of center + neighbors)
        fused = fuse_fields(x, left_v, right_v)  # (N,512)

        # 3. Pass fused vector through UniPhiOS engine
        bloom, identity_next, crown, triad, spiral = self.engine(fused)

        # identity_next is (N,512) and already layer-normalized
        new_state = identity_next

        # Update internal state
        self.state.data = new_state

        return {
            "state": new_state,
            "bloom": bloom,
            "crown": crown,
            "triad": triad,
            "spiral": spiral,
        }

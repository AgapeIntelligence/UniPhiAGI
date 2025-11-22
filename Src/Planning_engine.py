import torch
from torch import nn


class PlanningEngine(nn.Module):
    """
    PlanningEngine

    A simple deterministic reasoning layer:
    - Takes current lattice state
    - Optionally conditions on retrieved memory
    - Produces a candidate "intent" vector

    Architecture:
    intent = norm(W1 * state + W2 * memory)
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim

        self.state_proj = nn.Linear(dim, dim, bias=False)
        self.mem_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    @torch.no_grad()
    def forward(self, state: torch.Tensor, memory: torch.Tensor = None) -> torch.Tensor:
        h_state = self.state_proj(state)

        if memory is None:
            return self.norm(h_state)

        h_mem = self.mem_proj(memory)
        return self.norm(h_state + h_mem)

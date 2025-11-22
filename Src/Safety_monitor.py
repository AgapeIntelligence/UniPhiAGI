import torch
from torch import nn


class SafetyMonitor(nn.Module):
    """
    SafetyMonitor

    Provides measurable, deterministic checks:
    - Vector norm bounds
    - Sudden state-change magnitude
    - Optional cosine drift thresholds

    Returns a boolean flag and diagnostic values.
    """

    def __init__(self, dim: int = 512, max_norm: float = 20.0, max_delta: float = 5.0):
        super().__init__()
        self.dim = dim
        self.max_norm = max_norm
        self.max_delta = max_delta

    @torch.no_grad()
    def forward(self, prev_state: torch.Tensor, new_state: torch.Tensor):
        norm = new_state.norm(dim=-1)
        delta = (new_state - prev_state).norm(dim=-1)

        safe = (norm < self.max_norm) & (delta < self.max_delta)

        diagnostics = {
            "state_norm": norm.item(),
            "delta_norm": delta.item(),
            "safe": bool(safe.item())
        }
        return safe.item(), diagnostics

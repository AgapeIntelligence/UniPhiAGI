import torch
from torch import nn


class OutputAdapter(nn.Module):
    """
    OutputAdapter

    Converts a 512D internal state vector into:
    - Token logits (for decoding)
    - Or numerical output suitable for downstream APIs

    Deterministic, linear mapping.
    """

    def __init__(self, vocab_size: int = 32000, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    @torch.no_grad()
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.proj(state)  # (batch, vocab_size)

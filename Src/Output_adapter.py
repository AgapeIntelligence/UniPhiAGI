import torch
from torch import nn

class OutputAdapter(nn.Module):
    """
    OutputAdapter

    Converts a 512D internal state vector into:
    - Token logits (for decoding)
    - Or deterministic string output for demo purposes
    """

    def __init__(self, vocab_size: int = 32000, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    @torch.no_grad()
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # (batch, vocab_size)
        return self.proj(state)

    @torch.no_grad()
    def decode(self, state: torch.Tensor) -> str:
        """
        Simple deterministic decoding for demo:
        - argmax over vocab dimension
        - map to ASCII for readability
        """
        logits = self.forward(state)
        token_ids = logits.argmax(dim=-1).tolist()
        # handle batch dimension
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return "".join([chr(t % 128) for t in token_ids])

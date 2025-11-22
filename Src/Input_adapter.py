import torch
from torch import nn


class InputAdapter(nn.Module):
    """
    InputAdapter

    Converts tokenized text or numerical input into a fixed-size 512D vector.
    Deterministic design:
    - Token embedding (vocab_size x dim)
    - Mean-pooling
    - Optional linear compression

    This module assumes pre-tokenized input (list[int]).
    """

    def __init__(self, vocab_size: int = 32000, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.compress = nn.Linear(dim, dim, bias=False)

    @torch.no_grad()
    def forward(self, token_ids):
        tokens = torch.tensor(token_ids, dtype=torch.long)
        e = self.embedding(tokens)       # (T, dim)
        pooled = e.mean(dim=0, keepdim=True)
        return self.compress(pooled)

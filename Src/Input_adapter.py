import torch
from torch import nn

class InputAdapter(nn.Module):
    """
    InputAdapter

    Converts text or numerical input into a fixed-size 512D vector.
    Deterministic design:
    - Token embedding (vocab_size x dim)
    - Mean-pooling
    - Optional linear compression
    """

    def __init__(self, vocab_size: int = 32000, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.compress = nn.Linear(dim, dim, bias=False)

    @torch.no_grad()
    def forward(self, token_ids):
        """
        Convert list of integers (token IDs) to 512D vector.
        """
        if not isinstance(token_ids, torch.Tensor):
            tokens = torch.tensor(token_ids, dtype=torch.long)
        else:
            tokens = token_ids.long()
        e = self.embedding(tokens)       # (T, dim)
        pooled = e.mean(dim=0, keepdim=True)
        return self.compress(pooled)

    def encode(self, text: str):
        """
        Deterministic tokenization of text → token IDs → forward.
        Uses simple ordinal mapping for demo purposes.
        """
        token_ids = [ord(c) % self.embedding.num_embeddings for c in text]
        return self.forward(token_ids)

import torch
from torch import nn


class MemoryStore(nn.Module):
    """
    MemoryStore

    A persistent vector memory supporting:
    - Deterministic write (no stochastic consolidation)
    - Cosine-similarity retrieval
    - Optional top-k lookup

    Memory entries are stored as:
        keys:   (N, dim)
        values: (N, dim)
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.keys = []
        self.values = []

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor):
        key = key.detach().cpu()
        value = value.detach().cpu()
        self.keys.append(key)
        self.values.append(value)

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor, top_k: int = 1):
        if not self.keys:
            return None

        keys = torch.stack(self.keys)        # (N, dim)
        values = torch.stack(self.values)    # (N, dim)

        # Cosine similarity
        q = query / (query.norm(dim=-1, keepdim=True) + 1e-8)
        k = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)

        scores = (q * k).sum(dim=-1)         # (N,)

        if top_k == 1:
            idx = scores.argmax().item()
            return values[idx]

        # Top-k
        top_indices = torch.topk(scores, k=top_k).indices
        return values[top_indices]

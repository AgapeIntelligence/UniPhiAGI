"""
Zero-apotheosis normalization primitives.
"""

import torch


def zero_apotheosis_tensor(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize a tensor toward a small magnitude while preserving direction.
    Deterministic and numerically stable.
    """
    norm = t.norm(dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return t / norm * eps


def zero_apotheosis_scalar(x: float, eps: float = 1e-6) -> float:
    return float(eps if abs(x) < eps else x)

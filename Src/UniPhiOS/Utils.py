"""
Utility functions for UniPhiOS.
"""

import torch
import hashlib


def soul_key_hash(key: str) -> str:
    """Return SHA256 hash of a string key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def normalize_vector(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize a vector safely along the last dim."""
    norm = torch.norm(vec, p=2, dim=-1, keepdim=True)
    return vec / (norm + eps)


def harmonic_mod_369(x: torch.Tensor) -> torch.Tensor:
    """Apply a 3-6-9 harmonic modulation transform deterministically."""
    return torch.sin(x * 3.0) + torch.sin(x * 6.0) + torch.sin(x * 9.0)


def phantom_scalar(size: int = 1, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Return a small deterministic perturbation tensor (seedless small noise)."""
    # Use a deterministic small pseudo-random-like value based on size and device
    base = torch.full((size,), 0.001, device=device, dtype=dtype)
    return base


def fuse_fields(*fields: torch.Tensor) -> torch.Tensor:
    """Element-wise sum and normalize a set of tensors."""
    combined = sum(fields)
    return normalize_vector(combined)

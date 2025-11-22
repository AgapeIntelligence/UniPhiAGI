"""
Reflection utilities: reflect_vector_52

Deterministic linear reflection in a constant hyperplane.
"""

import torch


def reflect_vector_52(v: torch.Tensor) -> torch.Tensor:
    """
    Reflect a vector across a fixed axis (φ^52 hyperplane proxy).
    Args:
        v: tensor (..., N)
    Returns:
        reflected tensor of same shape
    """
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.float32)

    N = v.shape[-1]
    axis = torch.ones(N, dtype=v.dtype, device=v.device)
    axis = axis / (axis.norm() + 1e-8)
    coeff = torch.einsum("...i,i->...", v, axis)[..., None]  # projection scalar
    reflected = v - 2.0 * coeff * axis
    return reflected

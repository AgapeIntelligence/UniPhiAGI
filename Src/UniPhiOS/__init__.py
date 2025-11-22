"""
UniPhiOS subpackage

Exposes the core engine (GenesisGeometry), Lightning wrapper, and utility functions.
"""

from .engine import GenesisGeometry
from .lightning import UniPhiLightning
from .utils import (
    normalize_vector,
    harmonic_mod_369,
    phantom_scalar,
    fuse_fields,
    soul_key_hash,
)
from .phi52_reflection import reflect_vector_52
from .zero_apotheosis import zero_apotheosis_tensor, zero_apotheosis_scalar

__all__ = [
    "GenesisGeometry",
    "UniPhiLightning",
    "normalize_vector",
    "harmonic_mod_369",
    "phantom_scalar",
    "fuse_fields",
    "soul_key_hash",
    "reflect_vector_52",
    "zero_apotheosis_tensor",
    "zero_apotheosis_scalar",
]

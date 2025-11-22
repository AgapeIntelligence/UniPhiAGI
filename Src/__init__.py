"""
UniPhiAGI: Core modules for the full-stack AGI architecture.

This package exposes:
- Deterministic lattice engine (state update dynamics)
- Input/Output adapters (text ↔ vector)
- Planning engine (reasoning/decision layer)
- MemoryStore (persistent vector memory)
- SafetyMonitor (stability and coherence checks)
- UniPhiOS integration under src.uniphi_os
"""

from .lattice_engine import DeterministicLattice
from .memory_store import MemoryStore
from .planning_engine import PlanningEngine
from .input_adapter import InputAdapter
from .output_adapter import OutputAdapter
from .safety_monitor import SafetyMonitor

# Subpackage: UniPhiOS core
from . import uniphi_os

__all__ = [
    "DeterministicLattice",
    "MemoryStore",
    "PlanningEngine",
    "InputAdapter",
    "OutputAdapter",
    "SafetyMonitor",
    "uniphi_os",
]

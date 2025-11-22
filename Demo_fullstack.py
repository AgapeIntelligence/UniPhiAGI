"""
demo_fullstack.py
------------------
Demonstrates the end-to-end UniPhiAGI pipeline:

1. InputAdapter → 512-D embedding
2. UniPhiOS Engine → feature transform
3. DeterministicLattice (vΩ engine) → state update
4. PlanningEngine → action/response computation
5. MemoryStore → persistent retrieval/updates
6. OutputAdapter → textual/API output

This is a minimal scientific demo, not a benchmark.
"""

import torch

from Src.input_adapter import InputAdapter
from Src.output_adapter import OutputAdapter
from Src.lattice_engine import DeterministicLattice
from Src.planning_engine import PlanningEngine
from Src.memory_store import MemoryStore
from Src.safety_monitor import SafetyMonitor

# Integrated UniPhiOS core
from Src.uniphi_os.engine import GenesisGeometry


def run_demo(text: str = "Hello world"):
    """Runs a single forward pass through the UniPhiAGI stack."""

    # -----------------------
    # 1. Input → 512D vector
    # -----------------------
    input_adapter = InputAdapter()
    x = input_adapter.encode(text)   # torch.tensor [1, 512]

    # -----------------------
    # 2. UniPhiOS transform
    # -----------------------
    uniphi = GenesisGeometry(device="cpu", dtype=torch.float32)
    bloom, identity_next, *_ = uniphi(x)

    # -----------------------
    # 3. Lattice update
    # -----------------------
    lattice = DeterministicLattice(dim=512)
    lattice_state = lattice.forward(bloom)  # deterministic update using bloom

    # -----------------------
    # 4. Planning / reasoning
    # -----------------------
    planner = PlanningEngine()
    planned = planner.compute(lattice_state, identity_next)

    # -----------------------
    # 5. Memory integration
    # -----------------------
    memory = MemoryStore(dim=512)
    retrieved = memory.retrieve(planned)
    updated = memory.update(planned)

    # -----------------------
    # 6. Safety checks
    # -----------------------
    safety = SafetyMonitor()
    safe_output = safety.validate(updated)

    # -----------------------
    # 7. Output decode
    # -----------------------
    output_adapter = OutputAdapter()
    decoded = output_adapter.decode(safe_output)

    return {
        "input_vector": x,
        "uniphi_bloom": bloom,
        "lattice_state": lattice_state,
        "plan_vector": planned,
        "memory_retrieved": retrieved,
        "updated_state": updated,
        "final_output": decoded,
    }


if __name__ == "__main__":
    out = run_demo("Test input string for UniPhiAGI.")
    print("\n=== UniPhiAGI Demo Output ===")
    print(out["final_output"])

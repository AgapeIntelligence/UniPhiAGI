"""
UniPhiAGI Main Entry Point
--------------------------
Coordinates input → UniPhiOS → lattice update → planning → memory → safety → output
"""

import torch

from Src.input_adapter import InputAdapter
from Src.output_adapter import OutputAdapter
from Src.lattice_engine import DeterministicLattice
from Src.planning_engine import PlanningEngine
from Src.memory_store import MemoryStore
from Src.safety_monitor import SafetyMonitor
from Src.uniphi_os.engine import GenesisGeometry


def main(input_text: str = "Hello world"):
    # -----------------------
    # 1. Input Adapter
    # -----------------------
    input_adapter = InputAdapter()
    x = input_adapter.encode(input_text)  # [1, 512]

    # -----------------------
    # 2. UniPhiOS Engine
    # -----------------------
    uniphi = GenesisGeometry(device="cpu", dtype=torch.float32)
    bloom, identity_next, *_ = uniphi(x)

    # -----------------------
    # 3. Lattice Engine
    # -----------------------
    lattice = DeterministicLattice(dim=512)
    lattice_state = lattice.forward(bloom)  # deterministic update

    # -----------------------
    # 4. Planning Engine
    # -----------------------
    planner = PlanningEngine()
    plan_vector = planner.compute(lattice_state, identity_next)

    # -----------------------
    # 5. Memory Store
    # -----------------------
    memory = MemoryStore(dim=512)
    memory_retrieved = memory.retrieve(plan_vector)
    updated_state = memory.update(plan_vector)

    # -----------------------
    # 6. Safety Monitor
    # -----------------------
    safety = SafetyMonitor()
    safe_output = safety.validate(updated_state)

    # -----------------------
    # 7. Output Adapter
    # -----------------------
    output_adapter = OutputAdapter()
    final_output = output_adapter.decode(safe_output)

    # -----------------------
    # 8. Return results
    # -----------------------
    return {
        "input_vector": x,
        "uniphi_bloom": bloom,
        "lattice_state": lattice_state,
        "plan_vector": plan_vector,
        "memory_retrieved": memory_retrieved,
        "updated_state": updated_state,
        "final_output": final_output,
    }


if __name__ == "__main__":
    result = main("Test input string for UniPhiAGI.")
    print("\n=== UniPhiAGI Main Output ===")
    print(result["final_output"])

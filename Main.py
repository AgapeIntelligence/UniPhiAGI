"""
UniPhiAGI Main Entry Point
Coordinates input → lattice update → memory → output.
"""

import torch
from src.lattice_engine import DeterministicLattice
from src.input_adapter import InputAdapter
from src.output_adapter import OutputAdapter
from src.uniphi_os.engine import GenesisGeometry
from src.safety_monitor import SafetyMonitor

def main():
    # Core engine
    engine = GenesisGeometry(device="cpu", dtype=torch.float32)

    # Lattice
    lattice = DeterministicLattice(engine=engine)

    # Safety monitor
    safety = SafetyMonitor()

    # Adapters
    encode = InputAdapter()
    decode = OutputAdapter()

    # Example input vector
    x = torch.randn(1, 512)

    # Encode → Lattice → Memory → Output
    state = encode(x)
    node = lattice.step(state)

    if not safety.check(node):
        raise RuntimeError("Safety threshold exceeded")

    print(decode(node))

if __name__ == "__main__":
    main()

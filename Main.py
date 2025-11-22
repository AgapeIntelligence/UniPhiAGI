"""
UniPhiAGI Main Entry Point
Coordinates input → lattice update → memory → output.
"""

import torch
from Src.lattice_engine import DeterministicLattice
from Src.input_adapter import InputAdapter
from Src.output_adapter import OutputAdapter
from Src.uniphi_os.engine import GenesisGeometry
from Src.safety_monitor import SafetyMonitor

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

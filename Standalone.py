"""
Standalone UniPhiAGI Demo
-------------------------
Demonstrates the end-to-end pipeline without package imports.
"""

import torch
from torch import nn

# -----------------------
# Input Adapter
# -----------------------
class InputAdapter(nn.Module):
    def __init__(self, vocab_size: int = 32000, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.compress = nn.Linear(dim, dim, bias=False)

    @torch.no_grad()
    def encode(self, token_ids):
        tokens = torch.tensor(token_ids if isinstance(token_ids, list) else [1]*10, dtype=torch.long)
        e = self.embedding(tokens)
        pooled = e.mean(dim=0, keepdim=True)
        return self.compress(pooled)

# -----------------------
# Output Adapter
# -----------------------
class OutputAdapter(nn.Module):
    def __init__(self, vocab_size: int = 32000, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    @torch.no_grad()
    def decode(self, state):
        return self.proj(state)

# -----------------------
# UniPhiOS minimal engine
# -----------------------
class GenesisGeometry(nn.Module):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.linear = nn.Linear(512, 512, dtype=dtype).to(device)

    def forward(self, x):
        x = x.to(device=self.device, dtype=self.dtype)
        bloom = torch.tanh(self.linear(x))
        return bloom, x, bloom, bloom, bloom

# -----------------------
# Lattice Engine
# -----------------------
class DeterministicLattice:
    def __init__(self, dim=512):
        self.state = torch.zeros(1, dim)
    def forward(self, bloom):
        self.state = self.state + 0.1 * bloom
        return self.state

# -----------------------
# Planning Engine
# -----------------------
class PlanningEngine:
    def compute(self, lattice_state, identity_next):
        return 0.5 * lattice_state + 0.5 * identity_next

# -----------------------
# Memory Store
# -----------------------
class MemoryStore:
    def __init__(self, dim=512):
        self.memory = torch.zeros(1, dim)
    def retrieve(self, vector):
        return self.memory
    def update(self, vector):
        self.memory = 0.9 * self.memory + 0.1 * vector
        return self.memory

# -----------------------
# Safety Monitor
# -----------------------
class SafetyMonitor:
    def __init__(self, threshold=10.0):
        self.threshold = threshold
    def validate(self, vector):
        return torch.clamp(vector, -self.threshold, self.threshold)

# -----------------------
# Demo
# -----------------------
def run_demo(text="Hello world"):
    input_adapter = InputAdapter()
    x = input_adapter.encode(text)

    uniphi = GenesisGeometry()
    bloom, identity_next, *_ = uniphi(x)

    lattice = DeterministicLattice()
    lattice_state = lattice.forward(bloom)

    planner = PlanningEngine()
    plan_vector = planner.compute(lattice_state, identity_next)

    memory = MemoryStore()
    memory_retrieved = memory.retrieve(plan_vector)
    updated_state = memory.update(plan_vector)

    safety = SafetyMonitor()
    safe_output = safety.validate(updated_state)

    output_adapter = OutputAdapter()
    final_output = output_adapter.decode(safe_output)

    return final_output

if __name__ == "__main__":
    out = run_demo("Test input for UniPhiAGI.")
    print("\n=== UniPhiAGI Demo Output ===")
    print(out)

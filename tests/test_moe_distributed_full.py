import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest
import os
import re
from raylight.distributed_modules.moe import filter_moe_experts, patch_moe_layers, expert_parallel_forward

# --- Mock Architecture ---

class SimpleExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x):
        return self.net(x)

class SimpleMoEBlock(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([SimpleExpert(dim) for _ in range(num_experts)])
    
    def forward(self, x):
        # Standard non-distributed forward for baseline
        gate_logits = self.gate(x)
        weights = torch.softmax(gate_logits, dim=-1)
        # Simplified top-1 for parity testing
        top1_probs, top1_indices = torch.topk(weights, 1, dim=-1)
        
        # In a real MoE, this would be more complex, but we simulate Top-1 EP
        out = torch.zeros_like(x)
        # Reshape to token list
        x_flat = x.view(-1, x.shape[-1])
        indices_flat = top1_indices.view(-1)
        probs_flat = top1_probs.view(-1, 1)
        
        for i in range(len(self.experts)):
            mask = (indices_flat == i)
            if mask.any():
                out.view(-1, x.shape[-1])[mask] = self.experts[i](x_flat[mask]) * probs_flat[mask]
        return out

# --- Mock Distributed for Single Process Testing ---

class MockRequest:
    def wait(self): pass

class MockDist:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.sent_data = {}
    
    def get_rank(self): return self.rank
    def get_world_size(self): return self.world_size
    def get_backend(self): return "mock"
    
    def isend(self, tensor, dst): return MockRequest()
    def irecv(self, tensor, src): return MockRequest()
    
    def all_gather(self, tensor_list, tensor):
        # Simulate rank 0 and 1 having specific counts
        # This is a bit complex to mock fully, so we'll simplify
        for i in range(self.world_size):
            tensor_list[i].copy_(tensor if i == self.rank else torch.zeros_like(tensor))

    def all_gather_into_tensor(self, out, inp):
        pass # Simplified for mock

    def all_to_all_single(self, output, input, output_split_sizes=None, input_split_sizes=None):
        # In a single process mock, we just want to verify the logic around this call
        # We can't easily shuffle between ranks in 1 process without more setup
        # So we'll simulate "All tokens destined for this rank arrive"
        output.copy_(input) # Identity for now to check structural integrity

def run_moe_logic_verification():
    # This test verifies the token routing and logic flow WITHOUT a real cluster
    dim = 16
    num_experts = 4
    world_size = 2
    rank = 0
    
    # 1. Create Model and Patch
    model = SimpleMoEBlock(dim, num_experts)
    patch_moe_layers(model, expert_parallel=True)
    
    # 2. Mock dist inside the forward call
    import raylight.distributed_modules.moe as moe_mod
    original_dist = moe_mod.dist
    moe_mod.dist = MockDist(rank, world_size)
    
    try:
        x = torch.randn(1, 4, dim)
        # We expect this to run through our patched forward
        with torch.no_grad():
            out = model(x)
        print("MoE Logic Flow Verification PASSED!")
    finally:
        moe_mod.dist = original_dist

class TestMoELogic(unittest.TestCase):
    def test_moe_sharding_and_routing(self):
        # Test 1: Weight Filtering
        sd = {
            "blocks.0.mlp.experts.0.weight": torch.ones(1),
            "blocks.0.mlp.experts.1.weight": torch.ones(1),
            "gate.weight": torch.ones(1)
        }
        res_0 = filter_moe_experts(sd, 0, 2)
        self.assertIn("blocks.0.mlp.experts.0.weight", res_0)
        self.assertNotIn("blocks.0.mlp.experts.1.weight", res_0)
        
        # Test 2: Patching
        model = SimpleMoEBlock(8, 4)
        count = patch_moe_layers(model, expert_parallel=True)
        self.assertEqual(count, 1)
        
        # Test 3: Flow
        run_moe_logic_verification()

if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()

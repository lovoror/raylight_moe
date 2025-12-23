import torch
import torch.nn as nn
import torch.distributed as dist
import unittest
from raylight.distributed_modules.moe import filter_moe_experts, patch_moe_layers
import os

# Dummy MoE Block that looks like WanMoE
class DummyMoE(nn.Module):
    def __init__(self, num_experts=4, dim=64):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_experts)])

    def forward(self, x):
        gate_logits = self.gate(x)
        weights = torch.softmax(gate_logits, dim=-1)
        # Standard forward for comparison
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out += weights[..., i:i+1] * expert(x)
        return out

class TestMoEIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock distributed environment for single-process testing
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group("gloo", rank=0, world_size=1)

    def test_patching_and_forward(self):
        dim = 32
        num_experts = 4
        model = nn.Module()
        model.diffusion_model = nn.Module()
        model.diffusion_model.moe_block = DummyMoE(num_experts, dim)
        
        # 1. Test filtering
        sd = model.state_dict()
        # Mock keys in sd
        moe_sd = {
            "diffusion_model.moe_block.experts.0.weight": torch.randn(dim, dim),
            "diffusion_model.moe_block.experts.1.weight": torch.randn(dim, dim),
            "diffusion_model.moe_block.experts.2.weight": torch.randn(dim, dim),
            "diffusion_model.moe_block.experts.3.weight": torch.randn(dim, dim),
            "diffusion_model.moe_block.gate.weight": torch.randn(num_experts, dim),
        }
        
        filtered_sd = filter_moe_experts(moe_sd, 0, 1) # Rank 0, World Size 1
        # Should keep all experts since world_size=1
        self.assertEqual(len([k for k in filtered_sd if "experts" in k]), 4)

        # 2. Test patching
        patched_count = patch_moe_layers(model.diffusion_model, expert_parallel=True)
        self.assertEqual(patched_count, 1)
        self.assertTrue(hasattr(model.diffusion_model.moe_block, "original_forward"))
        
        # 3. Test forward pass execution
        x = torch.randn(1, 10, dim)
        try:
            output = model.diffusion_model.moe_block(x)
            self.assertEqual(output.shape, x.shape)
            print("MoE Forward Integration Test Passed!")
        except Exception as e:
            self.fail(f"MoE Block forward failed: {e}")

if __name__ == "__main__":
    unittest.main()

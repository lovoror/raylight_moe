import torch
import torch.nn as nn
import unittest
import os
from raylight.distributed_modules.moe import filter_moe_experts, patch_moe_layers
from raylight.diffusion_models.wan.fsdp import shard_model_fsdp2

# --- Test Case ---
class TestMoEUltimate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use simple Gloo group for single-process verification
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12357'
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("gloo", rank=0, world_size=1)

    def test_all_logics(self):
        print("\n[Test] Starting Ultimate MoE Logic Verification...")
        
        # 1. Test Weight Filtering
        sd = {
            "blocks.0.mlp.experts.0.weight": torch.randn(1),
            "blocks.0.mlp.experts.1.weight": torch.randn(1)
        }
        # In a 1-rank world, we keep expert 0 (0%1 == 0) and expert 1 (1%1 == 0)
        res = filter_moe_experts(sd, 0, 1)
        self.assertIn("blocks.0.mlp.experts.0.weight", res)
        self.assertIn("blocks.0.mlp.experts.1.weight", res)
        print("  - [OK] Weight Filtering (Selective Loading)")

        # 2. Test FSDP Sharding Exclusion
        # We need to mock fully_shard because we are in a single process
        import raylight.diffusion_models.wan.fsdp as fsdp_mod
        from unittest import mock
        
        model = nn.Module()
        model.diffusion_model = nn.Module()
        model.diffusion_model.blocks = nn.ModuleList([nn.Module()])
        model.diffusion_model.blocks[0].mlp = nn.Module()
        model.diffusion_model.blocks[0].mlp.experts = nn.ModuleList([nn.Parameter(torch.randn(1))])
        # Add a dummy attn to satisfy detect_dtype_mismatch
        model.diffusion_model.blocks[0].self_attn = nn.Module()
        model.diffusion_model.blocks[0].self_attn.v = nn.Module()
        model.diffusion_model.blocks[0].self_attn.v.weight = nn.Parameter(torch.randn(1))

        ignored_in_fsdp = []
        with mock.patch("raylight.diffusion_models.wan.fsdp.fully_shard") as mock_fsdp:
            def side_effect(module, **kwargs):
                if 'ignored_params' in kwargs:
                    ignored_in_fsdp.extend(kwargs['ignored_params'])
                return module
            mock_fsdp.side_effect = side_effect
            
            shard_model_fsdp2(model, {}, False, expert_parallel=True)
            expert_params = [p for n, p in model.named_parameters() if "experts" in n]
            all_ignored = all(any(p is ip for ip in ignored_in_fsdp) for p in expert_params)
            self.assertTrue(all_ignored, "Experts were not ignored by FSDP!")
            print("  - [OK] FSDP Expert Isolation")

        # 3. Test Patching & Forward Flow
        # Use our real MoE logic but in a 1-rank world
        class MockMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(8, 4)
                self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
            def forward(self, x):
                return x

        moe_block = MockMoE()
        patch_moe_layers(moe_block, expert_parallel=True)
        self.assertTrue(hasattr(moe_block, "original_forward"))
        
        x = torch.randn(1, 4, 8)
        with torch.no_grad():
            out = moe_block(x)
        self.assertEqual(out.shape, x.shape)
        print("  - [OK] Dynamic Patching & Forward Flow (1-Rank)")

        print("[SUCCESS] All MoE Expert Parallelism Logics Verified!")

if __name__ == "__main__":
    unittest.main()

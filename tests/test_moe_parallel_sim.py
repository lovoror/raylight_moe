import torch
import unittest
from raylight.distributed_modules.moe import filter_moe_experts

class TestMoEParallel(unittest.TestCase):
    def test_filter_moe_experts(self):
        # Create a mock state_dict
        sd = {
            "blocks.0.mlp.experts.0.weight": torch.randn(10, 10),
            "blocks.0.mlp.experts.1.weight": torch.randn(10, 10),
            "blocks.0.mlp.experts.2.weight": torch.randn(10, 10),
            "blocks.0.mlp.experts.3.weight": torch.randn(10, 10),
            "blocks.0.mlp.gate.weight": torch.randn(10, 10),
            "other_layer.weight": torch.randn(10, 10)
        }
        
        # Simulating 2 ranks
        world_size = 2
        
        # Rank 0 should keep experts 0, 2
        sd_0 = filter_moe_experts(sd.copy(), 0, world_size)
        self.assertIn("blocks.0.mlp.experts.0.weight", sd_0)
        self.assertIn("blocks.0.mlp.experts.2.weight", sd_0)
        self.assertNotIn("blocks.0.mlp.experts.1.weight", sd_0)
        self.assertNotIn("blocks.0.mlp.experts.3.weight", sd_0)
        self.assertIn("blocks.0.mlp.gate.weight", sd_0)
        
        # Rank 1 should keep experts 1, 3
        sd_1 = filter_moe_experts(sd.copy(), 1, world_size)
        self.assertIn("blocks.0.mlp.experts.1.weight", sd_1)
        self.assertIn("blocks.0.mlp.experts.3.weight", sd_1)
        self.assertNotIn("blocks.0.mlp.experts.0.weight", sd_1)
        self.assertNotIn("blocks.0.mlp.experts.2.weight", sd_1)
        self.assertIn("blocks.0.mlp.gate.weight", sd_1)
        
        print("MoE Filter Test Passed!")

if __name__ == "__main__":
    unittest.main()

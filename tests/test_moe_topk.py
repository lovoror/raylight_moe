import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from raylight.distributed_modules.moe import expert_parallel_forward

class MockExpert(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        # Fix: Add a dummy parameter to avoid "module has no parameters" issues
        self.weight = torch.nn.Parameter(torch.ones(1) * factor)

    def forward(self, x):
        return x * self.weight

class MockMoE(torch.nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.experts = torch.nn.ModuleList([MockExpert(i + 1.0) for i in range(num_experts)])

def run_test(rank, world_size):
    # Initialize process group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29505'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Setup
    num_experts = 4
    hidden_dim = 8
    batch_size = 1
    seq_len = 2
    top_k = 2
    
    device = torch.device("cpu")
    model = MockMoE(num_experts).to(device)
    
    # Input x: [1, 2, 8]
    x = torch.ones(batch_size, seq_len, hidden_dim, device=device)
    
    # Gate logits: tokens will pick experts
    # Token 0: pick expert 0 and 1
    # Token 1: pick expert 2 and 3
    gate_logits = torch.zeros(batch_size, seq_len, num_experts, device=device)
    gate_logits[0, 0, 0] = 10.0
    gate_logits[0, 0, 1] = 5.0
    gate_logits[0, 1, 2] = 10.0
    gate_logits[0, 1, 3] = 5.0
    
    # Expected probs for Token 0 (Top-2): 
    # softmax([10, 5]) -> [0.993, 0.007] approx
    probs = torch.softmax(gate_logits, dim=-1)
    topk_probs, _ = torch.topk(probs, top_k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    
    # Run forward
    output = expert_parallel_forward(model, x, gate_logits, top_k=top_k)
    
    # Verification
    print(f"Rank {rank} Output Shape: {output.shape}")
    
    # Token 0 weights: exp0 (factor 1), exp1 (factor 2)
    # Expected Token 0 = 1.0 * (exp0_weight * p0 + exp1_weight * p1)
    # Token 1 weights: exp2 (factor 3), exp3 (factor 4)
    
    w0_0, w0_1 = topk_probs[0, 0]
    expected_val_0 = (1.0 * w0_0 + 2.0 * w0_1)
    
    w1_0, w1_1 = topk_probs[0, 1]
    expected_val_1 = (3.0 * w1_0 + 4.0 * w1_1)
    
    # Check values (allowing for small float epsilon)
    diff0 = torch.abs(output[0, 0, 0] - expected_val_0).item()
    diff1 = torch.abs(output[0, 1, 0] - expected_val_1).item()
    
    success = diff0 < 1e-4 and diff1 < 1e-4
    
    with open("/tmp/moe_test_result.txt", "a") as f:
        if success:
            f.write(f"Rank {rank}: Test PASSED ✅\n")
        else:
            f.write(f"Rank {rank}: Test FAILED ❌ (diff0={diff0:.6f}, diff1={diff1:.6f})\n")
            f.write(f"Rank {rank}: Got {output[0, 0, 0].item()}, expected {expected_val_0}\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    if os.path.exists("/tmp/moe_test_result.txt"):
        os.remove("/tmp/moe_test_result.txt")
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
    with open("/tmp/moe_test_result.txt", "r") as f:
        print(f.read())

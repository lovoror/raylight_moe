import torch
from torch import distributed as dist
import re
import logging

def filter_moe_experts(sd, rank, world_size):
    """
    Filters the state_dict so each rank only keeps its assigned experts.
    Assumes expert layers follow the pattern: blocks.<block_idx>.mlp.experts.<expert_idx>.<param_name>
    """
    # Pattern to find experts in Wan2.2 (and many others)
    # e.g., blocks.0.mlp.experts.0.weight
    expert_pattern = re.compile(r"blocks\.\d+\.mlp\.experts\.(\d+)\..*")
    
    # First, find how many experts per block
    block_experts = {}
    for key in sd.keys():
        match = expert_pattern.match(key)
        if match:
            # Extract block index from key
            block_idx = int(key.split('.')[1])
            expert_idx = int(match.group(1))
            if block_idx not in block_experts:
                block_experts[block_idx] = set()
            block_experts[block_idx].add(expert_idx)

    if not block_experts:
        logging.info("[MoE] No expert layers detected in state_dict.")
        return sd

    # Determine which experts this rank should keep
    # For simplicity, we use a global expert index or per-block?
    # Usually experts are sharded globally across the expert-parallel group.
    new_sd = {}
    total_filtered = 0
    total_kept = 0
    
    for key, value in sd.items():
        match = expert_pattern.match(key)
        if match:
            expert_idx = int(match.group(1))
            # Expert Parallelism: Rank i keeps experts where idx % world_size == i
            if expert_idx % world_size == rank:
                new_sd[key] = value
                total_kept += 1
            else:
                total_filtered += 1
                continue
        else:
            new_sd[key] = value

    logging.info(f"[MoE] Rank {rank}: Kept {total_kept} expert parameters, filtered {total_filtered}.")
    return new_sd

def safe_all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None):
    """
    Handles all_to_all_single with a fallback for backends (like Gloo on Windows) 
    that might not support it natively.
    """
    backend = dist.get_backend()
    if backend == "nccl" or (backend == "gloo" and hasattr(dist, "all_to_all_single")):
        try:
            return dist.all_to_all_single(output, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes)
        except Exception as e:
            logging.debug(f"[MoE] Native all_to_all_single failed, falling back: {e}")
    
    # Fallback using all_gather
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # We need to know the max size to pad for all_gather if sizes are unequal
    # But wait, for many MoE cases, we can use a more efficient send/recv loop if all_to_all is missing
    # For Gloo, send/recv is usually reliable.
    
    input_splits = torch.split(input, input_split_sizes) if input_split_sizes else torch.chunk(input, world_size)
    output_splits = torch.split(output, output_split_sizes) if output_split_sizes else torch.chunk(output, world_size)
    
    reqs = []
    for i in range(world_size):
        if i == rank:
            output_splits[i].copy_(input_splits[i])
        else:
            # Send to i, receive from i
            reqs.append(dist.isend(input_splits[i], dst=i))
            reqs.append(dist.irecv(output_splits[i], src=i))
            
    for r in reqs:
        r.wait()

@torch.compiler.disable
def expert_parallel_forward(module, x, gate_logits):
    """
    Expert Parallel Forward pass.
    module: The MoE module being patched (contains self.experts)
    x: [Batch, Sequence, Hidden]
    gate_logits: [Batch, Sequence, NumExperts]
    """
    orig_shape = x.shape
    x = x.view(-1, orig_shape[-1])
    gate_logits = gate_logits.view(-1, gate_logits.shape[-1])

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_experts = gate_logits.shape[-1]
    experts_per_rank = num_experts // world_size

    # 1. Routing (Top-1 for simplicity, can be extended to Top-K)
    probs = torch.softmax(gate_logits, dim=-1)
    top1_probs, top1_indices = torch.topk(probs, 1, dim=-1)

    # 2. Prepare tokens to send
    # We need to sort tokens by their target rank
    target_ranks = top1_indices.flatten() // experts_per_rank
    
    # Count how many tokens go to each rank
    send_counts = torch.zeros(world_size, dtype=torch.long, device=x.device)
    for r in range(world_size):
        send_counts[r] = (target_ranks == r).sum()

    # Communication: tell other ranks how many tokens to expect
    recv_counts = torch.empty_like(send_counts)
    dist.all_gather_into_tensor(recv_counts.view(1, -1).expand(world_size, -1).contiguous(), send_counts)
    # Actually, all_gather on a single tensor is easier:
    all_send_counts = [torch.zeros_like(send_counts) for _ in range(world_size)]
    dist.all_gather(all_send_counts, send_counts)
    recv_counts = torch.stack([all_send_counts[i][rank] for i in range(world_size)])

    # Sort tokens for all_to_all
    sort_indices = torch.argsort(target_ranks)
    sorted_x = x[sort_indices]

    # 3. All-to-All Transfer
    recv_tokens = torch.empty(recv_counts.sum(), x.shape[-1], device=x.device, dtype=x.dtype)
    safe_all_to_all_single(recv_tokens, sorted_x, output_split_sizes=recv_counts.tolist(), input_split_sizes=send_counts.tolist())

    # 4. Local Expert Computation
    # We need to tell the receiving side which expert each token is for
    # Communication: send the expert indices too
    recv_expert_indices = torch.empty(recv_counts.sum(), dtype=torch.long, device=x.device)
    # We need to sort indices just like we sorted x
    sorted_expert_indices = top1_indices.flatten()[sort_indices]
    safe_all_to_all_single(recv_expert_indices, sorted_expert_indices, output_split_sizes=recv_counts.tolist(), input_split_sizes=send_counts.tolist())

    # Compute on local experts
    combined_output = torch.zeros_like(recv_tokens)
    
    for i, expert in enumerate(module.experts):
        # We only execute the experts assigned to this rank during loading
        if i % world_size == rank:
            mask = (recv_expert_indices == i)
            if mask.any():
                expert_input = recv_tokens[mask]
                expert_output = expert(expert_input)
                combined_output[mask] = expert_output

    # 5. All-to-All Back
    final_sorted_output = torch.empty_like(sorted_x)
    safe_all_to_all_single(final_sorted_output, combined_output, output_split_sizes=send_counts.tolist(), input_split_sizes=recv_counts.tolist())

    # 6. Reorder back to original sequence
    output = torch.empty_like(x)
    output[sort_indices] = final_sorted_output
    
    # Apply gating probability
    output = output * top1_probs.view(-1, 1)

    return output.view(orig_shape)

def patch_moe_layers(model, expert_parallel=False):
    """
    Automatically finds and patches MoE layers in the model.
    """
    import types
    if not expert_parallel:
        return 0
        
    patched_count = 0
    for name, module in model.named_modules():
        # Heuristic to find MoE layers: contains 'experts' ModuleList and a 'gate' or 'router'
        has_experts = hasattr(module, "experts") and isinstance(module.experts, torch.nn.ModuleList)
        has_gate = hasattr(module, "gate") or hasattr(module, "router")
        
        if has_experts and has_gate:
            if not hasattr(module, "original_forward"):
                print(f"[MoE] Patching MoE layer: {name}")
                module.original_forward = module.forward
                
                # We need to capture the current module in a closure
                def make_patched_forward(m):
                    def new_forward(self, x, *args, **kwargs):
                        # Most MoE blocks in DiT/Transformer expect (x, ...)
                        # We need to intercept the gate computation or the expert execution.
                        # Since we want to shuffle tokens, we must intercept BEFORE expert execution.
                        
                        # In many implementations, the forward looks like:
                        # gate_logits = self.gate(x)
                        # ...
                        # return self.expert_parallel_forward(x, gate_logits)
                        
                        # If we patch the WHOLE forward, we might miss some context.
                        # A better way is to patch the internal expert execution if possible,
                        # but that's very implementation dependent.
                        
                        # Generic approach:
                        # 1. Compute gate_logits
                        if hasattr(m, "gate"):
                            gate_logits = m.gate(x)
                        else:
                            gate_logits = m.router(x)
                        
                        # 2. Call our distributed forward
                        return expert_parallel_forward(m, x, gate_logits)
                    
                    return new_forward

                module.forward = types.MethodType(make_patched_forward(module), module)
                patched_count += 1
    
    if patched_count > 0:
        logging.info(f"[MoE] Successfully patched {patched_count} MoE layers for Expert Parallelism.")
    return patched_count

class MoERouter(torch.nn.Module):
    # (TODO) Implement distributed routing logic
    pass

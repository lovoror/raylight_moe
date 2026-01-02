"""
Diagnostic wrapper to intercept and log CFG parallel forward calls.
This will help us understand what arguments are actually being passed.
"""
import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def cfg_parallel_forward_wrapper_diagnostic(executor, *args, **kwargs):
    """Diagnostic version that logs all inputs"""
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()
    
    # Log everything we receive
    print(f"\n{'='*80}")
    print(f"[CFG DIAGNOSTIC] Rank {cfg_rank}/{cfg_world_size}")
    print(f"[CFG DIAGNOSTIC] Number of positional args: {len(args)}")
    print(f"[CFG DIAGNOSTIC] Kwargs keys: {list(kwargs.keys())}")
    
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            print(f"[CFG DIAGNOSTIC] args[{i}]: Tensor shape={arg.shape}, dtype={arg.dtype}")
        elif isinstance(arg, (list, tuple)):
            print(f"[CFG DIAGNOSTIC] args[{i}]: {type(arg).__name__} len={len(arg)}")
            for j, item in enumerate(arg):
                if isinstance(item, torch.Tensor):
                    print(f"[CFG DIAGNOSTIC]   [{j}]: Tensor shape={item.shape}")
        elif isinstance(arg, dict):
            print(f"[CFG DIAGNOSTIC] args[{i}]: dict keys={list(arg.keys())}")
        else:
            print(f"[CFG DIAGNOSTIC] args[{i}]: {type(arg).__name__} = {arg if not isinstance(arg, torch.Tensor) else '...'}")
    
    print(f"{'='*80}\n")
    
    # Call the original logic (simplified for now)
    # Just pass through without chunking to see what happens
    result = executor(*args, **kwargs)
    
    if isinstance(result, torch.Tensor):
        print(f"[CFG DIAGNOSTIC] Result shape: {result.shape}")
    
    # In CFG mode, we should gather results
    if cfg_world_size > 1:
        result = get_cfg_group().all_gather(result, dim=0)
    
    return result

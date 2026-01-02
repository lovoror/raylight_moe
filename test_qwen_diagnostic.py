#!/usr/bin/env python3
"""
Minimal test script to diagnose Qwen Edit 2511 CFG wrapper issues.
This will run a single forward pass and log all arguments.
"""
import os
import sys

# Setup paths
sys.path.insert(0, '/data/ComfyUI/custom_nodes/raylight/src')
sys.path.insert(0, '/data/ComfyUI/custom_nodes')
sys.path.insert(0, '/data/ComfyUI')

import torch
import gc

print(">>> Qwen Edit 2511 Diagnostic Test")
print("=" * 80)

# Import after path setup
import comfy.model_management
import comfy.utils
import comfy.sd

# Load model
model_path = "/data/ComfyUI/models/diffusion_models/qwen/qwen_image_edit_2511_fp8mixed.safetensors"
print(f"Loading model: {model_path}")

sd = comfy.utils.load_torch_file(model_path)
model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"dtype": torch.float16})

del sd
gc.collect()
torch.cuda.empty_cache()

print(f"Model type: {type(model.model).__name__}")
print(f"Model class: {model.model.__class__.__name__}")

# Check if CFG wrapper is applied
if hasattr(model, 'wrappers'):
    print(f"Wrappers: {model.wrappers}")
else:
    print("No wrappers attribute found")

# Apply CFG wrapper manually to test
from raylight.distributed_modules.cfg import CFGParallelInjectRegistry

print("\nApplying CFG wrapper...")
wrapper_func = CFGParallelInjectRegistry.inject(model)
print(f"Wrapper function: {wrapper_func.__name__}")

# Now test a forward pass
print("\n" + "=" * 80)
print("Testing forward pass...")
print("=" * 80)

# Create test inputs matching Qwen Edit
x = torch.zeros(1, 4, 64, 64).cuda().half()
timesteps = torch.zeros(1).cuda().half()
context = torch.zeros(1, 256, 3072).cuda().half()
attention_mask = torch.ones(1, 256).cuda().half()
ref_latents = [torch.zeros(1, 4, 64, 64).cuda().half()]
additional_t_cond = torch.zeros(1, 1280).cuda().half()
transformer_options = {}

# Get the actual diffusion model
diffusion_model = model.model.diffusion_model

print(f"\nDiffusion model type: {type(diffusion_model).__name__}")
print(f"Diffusion model has _forward: {hasattr(diffusion_model, '_forward')}")
print(f"Diffusion model has forward: {hasattr(diffusion_model, 'forward')}")

# Try calling the model directly
print("\n--- Direct model call (no wrapper) ---")
try:
    with torch.no_grad():
        result = diffusion_model(
            x, timesteps, context,
            attention_mask=attention_mask,
            ref_latents=ref_latents,
            additional_t_cond=additional_t_cond,
            transformer_options=transformer_options
        )
    print(f"SUCCESS: Output shape = {result.shape}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Now try with wrapper
print("\n--- Wrapped model call ---")
try:
    with torch.no_grad():
        # The wrapper expects: executor, *args, **kwargs
        # where args should be the positional arguments to the model
        result = wrapper_func(
            diffusion_model,
            x, timesteps, context, attention_mask, ref_latents, additional_t_cond, transformer_options
        )
    print(f"SUCCESS: Output shape = {result.shape}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Diagnostic test complete")

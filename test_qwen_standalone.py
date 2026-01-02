#!/usr/bin/env python3
"""
Standalone diagnostic test for Qwen Edit 2511 CFG wrapper.
This bypasses raylight package imports to avoid ModuleNotFoundError.
"""
import os
import sys
import torch
import gc

# Add ComfyUI to path
sys.path.insert(0, '/data/ComfyUI')

print(">>> Standalone Qwen Edit 2511 Diagnostic Test")
print("=" * 80)

# Import ComfyUI modules
import comfy.model_management
import comfy.utils
import comfy.sd

# Load model
model_path = "/data/ComfyUI/models/diffusion_models/qwen/qwen_image_edit_2511_fp8mixed.safetensors"
print(f"Loading model: {model_path}")

sd = comfy.utils.load_torch_file(model_path)
model_patcher = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"dtype": torch.float16})

del sd
gc.collect()
torch.cuda.empty_cache()

print(f"Model type: {type(model_patcher.model).__name__}")
print(f"Diffusion model type: {type(model_patcher.model.diffusion_model).__name__}")

# Get the diffusion model
diffusion_model = model_patcher.model.diffusion_model

# Check model signature by inspecting forward method
import inspect
forward_sig = inspect.signature(diffusion_model.forward if hasattr(diffusion_model, 'forward') else diffusion_model._forward)
print(f"\nModel forward signature: {forward_sig}")

# Create test inputs
print("\n" + "=" * 80)
print("Creating test inputs...")
x = torch.zeros(1, 4, 64, 64).cuda().half()
timesteps = torch.zeros(1).cuda().half()
context = torch.zeros(1, 256, 3072).cuda().half()
attention_mask = torch.ones(1, 256).cuda().half()
ref_latents = [torch.zeros(1, 4, 64, 64).cuda().half()]
additional_t_cond = torch.zeros(1, 1280).cuda().half()
transformer_options = {}

print(f"x shape: {x.shape}")
print(f"timesteps shape: {timesteps.shape}")
print(f"context shape: {context.shape}")
print(f"attention_mask shape: {attention_mask.shape}")
print(f"ref_latents: list of {len(ref_latents)} tensors, first shape: {ref_latents[0].shape}")
print(f"additional_t_cond shape: {additional_t_cond.shape}")

# Test direct call
print("\n" + "=" * 80)
print("Testing direct model call...")
print("=" * 80)

try:
    with torch.no_grad():
        result = diffusion_model(
            x, timesteps, context,
            attention_mask=attention_mask,
            ref_latents=ref_latents,
            additional_t_cond=additional_t_cond,
            transformer_options=transformer_options
        )
    print(f"✓ SUCCESS: Output shape = {result.shape}")
    print(f"✓ Output dtype = {result.dtype}")
    print(f"✓ Output device = {result.device}")
    
    # Check if output is all zeros/noise
    output_mean = result.mean().item()
    output_std = result.std().item()
    print(f"✓ Output mean = {output_mean:.6f}")
    print(f"✓ Output std = {output_std:.6f}")
    
    if abs(output_mean) < 1e-5 and output_std < 1e-5:
        print("⚠ WARNING: Output appears to be all zeros!")
    elif output_std > 100:
        print("⚠ WARNING: Output has very high variance (possible noise)!")
    else:
        print("✓ Output appears normal")
        
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Diagnostic test complete")

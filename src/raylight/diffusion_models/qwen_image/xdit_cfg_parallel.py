import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    # Signature should match usp_dit_forward:
    # x, timesteps, context, attention_mask, ref_latents, additional_t_cond, transformer_options
    if len(args) < 7:
        # Fallback for different versions or unexpected number of positional args
        x = args[0]
        timestep = args[1]
        context = args[2]
        attention_mask = kwargs.get('attention_mask', None) if len(args) < 4 else args[3]
        ref_latents = kwargs.get('ref_latents', None) if len(args) < 5 else args[4]
        additional_t_cond = kwargs.get('additional_t_cond', None) if len(args) < 6 else args[5]
        transformer_options = kwargs.get('transformer_options', {}) if len(args) < 7 else args[6]
    else:
        x, timestep, context, attention_mask, ref_latents, additional_t_cond, transformer_options = args

    is_chunked = False
    if x.shape[0] == cfg_world_size:
        x = torch.chunk(x, cfg_world_size, dim=0)[cfg_rank]
        is_chunked = True
    else:
        # Avoid error if x is already chunked or in single-batch mode
        pass
        
    def safe_chunk(tensor):
        if is_chunked and tensor is not None and isinstance(tensor, torch.Tensor) and tensor.shape[0] >= cfg_world_size:
            return torch.chunk(tensor, cfg_world_size, dim=0)[cfg_rank]
        return tensor

    timestep = safe_chunk(timestep)
    context = safe_chunk(context)
    attention_mask = safe_chunk(attention_mask)
    additional_t_cond = safe_chunk(additional_t_cond)

    if ref_latents is not None:
        if isinstance(ref_latents, (list, tuple)):
            ref_latents = [safe_chunk(r) for r in ref_latents]
        else:
            ref_latents = safe_chunk(ref_latents)

    result = executor(x, timestep, context, attention_mask, ref_latents, additional_t_cond, transformer_options, **kwargs)
    result = get_cfg_group().all_gather(result, dim=0)
    return result

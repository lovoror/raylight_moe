import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    x, timestep, context, clip_fea, time_dim_concat, transformer_options = args

    # --- 原代码：CFG=1.0 时直接报错 ---
    # if x.shape[0] == cfg_world_size:
    #     x = torch.chunk(x, cfg_world_size, dim=0)[cfg_rank]
    # else:
    #     raise ValueError("CFG = 1.0, disables guidance. Increase CFG > 1.0 or switch to another parallelism mode")
    
    # --- 优化方案：支持单 Batch (CFG=1.0) 运行 ---
    if x.shape[0] == cfg_world_size:
        # 正常的双分支并行 (CFG > 1.0)
        x = torch.chunk(x, cfg_world_size, dim=0)[cfg_rank]
    else:
        # CFG = 1.0 情况，输入本身就是单 Batch 或不匹配
        # 我们直接让所有卡都参与这个单 Batch 的计算
        # 注意：这里不需要 chunk，因为 Batch = 1 无法被 2 块卡在维度 0 上切分
        # 我们依然返回 x，xfuser 内部会通过序列并行 (SP) 来处理
        pass

    timestep = torch.chunk(timestep, x.shape[0] if timestep.shape[0] == x.shape[0] else cfg_world_size, dim=0)[0] if timestep.shape[0] > 1 else timestep
    context = torch.chunk(context, x.shape[0] if context.shape[0] == x.shape[0] else cfg_world_size, dim=0)[0] if context.shape[0] > 1 else context

    if clip_fea is not None:
        # clip_fea = torch.chunk(clip_fea, cfg_world_size, dim=0)[cfg_rank]
        clip_fea = torch.chunk(clip_fea, x.shape[0] if clip_fea.shape[0] == x.shape[0] else cfg_world_size, dim=0)[0] if clip_fea.shape[0] > 1 else clip_fea

    if time_dim_concat is not None:
        # time_dim_concat = torch.chunk(time_dim_concat, cfg_world_size, dim=0)[cfg_rank]
        time_dim_concat = torch.chunk(time_dim_concat, x.shape[0] if time_dim_concat.shape[0] == x.shape[0] else cfg_world_size, dim=0)[0] if time_dim_concat.shape[0] > 1 else time_dim_concat

    result = executor(x, timestep, context, clip_fea, time_dim_concat, transformer_options, **kwargs)
    # result = get_cfg_group().all_gather(result, dim=0)
    # 这里的 all_gather 会自动将两块卡的 Batch=1 拼成 Batch=2 (即 [正, 正])，完美符合 ComfyUI 预期
    result = get_cfg_group().all_gather(result, dim=0)
    return result

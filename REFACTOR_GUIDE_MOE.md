# Wan2.2 MoE 专家并行 (Expert Parallelism) 技术实现说明

## 1. 核心目标
解决 Wan2.2 (27B MoE) 模型在 V100 16G 显卡上使用 FSDP 时出现的 98% 显存峰值问题。通过**由外而内（权重分片）**和**由内而外（Token 交换）**的策略，实现真正的显存负载均衡。

## 2. 代码变更详细清单

### A. 通信内核 `src/raylight/distributed_modules/moe.py` [新增]
*   **功能**: EP 的“发动机”。
*   **权重过滤 (`filter_moe_experts`)**: 使用正则表达式扫描 `state_dict`，根据当前 GPU 的 Rank 进行专家分片。确保每张卡持有的专家数量 = 1/WorldSize。
*   **分布式协议 (`safe_all_to_all_single`)**: 核心通信层。针对 **NCCL (NVLink)** 优先使用硬件加速；针对 Windows 提供 Gloo P2P 回退机制。
*   **转发逻辑 (`expert_parallel_forward`)**: 实现了 Top-1 路由、Token Dispatch、All-to-All 数据交换和结果聚合。

### B. 节点界面 `src/raylight/nodes.py` [修改]
*   **新增 `Ray MoE Init Actor` 节点**: 
    *   继承自原有的初始化节点。
    *   新增 `expert_parallel` 布尔开关，作为全局控制位的源头。

### C. 模型加载 `src/raylight/distributed_worker/ray_worker.py` [修改]
*   **拦截加载流**: 在 `load_unet` 方法中接入了 `filter_moe_experts` 钩子。
*   **逻辑**: 权重在进入显存前就被截断，显存占用从模型加载那一刻起就是优化后的。

### D. FSDP 引擎适配 `src/raylight/comfy_dist/fsdp_registry.py` & `src/raylight/diffusion_models/wan/fsdp.py` [修改]
*   **参数传递**: 更新了 FSDP 注册表的签名，将 `expert_parallel` 信号透传至底层 sharding 函数。
*   **参数隔离 (Exclusion)**: 在 `shard_model_fsdp2` 中加入了识别逻辑。当 EP 开启时，FSDP 强制**忽略**所有专家模块（`mlp.experts.*`）。
*   **意义**: 防止 FSDP 重复切分专家权重，避免了在分布式屏障（Barrier）处发生同步显存高峰。

### E. 自动注入 `src/raylight/distributed_modules/usp.py` [修改]
*   **无感知集成**: 在 `USPInjectRegistry.inject` 中接入 `patch_moe_layers`。
*   **逻辑**: 在序列并行初始化的同时，扫描并替换模型内部的 MoE MLP 层。

## 3. 架构优势（针对 Ubuntu + NVLink）
1.  **NCCL 原生优化**: NVLink 环境下，`all_to_all` 通信几乎不占用 CPU，且带宽远超 PCIe。
2.  **物理显存隔离**: 专家权重不再在卡间动态迁移，而是静态驻留。这使得 VRAM 曲线极其平稳，不再有“锯齿状”尖峰。
3.  **兼容性**: 该架构理论上支持所有具有 `ModuleList` 类型专家的 MoE 模型（如 DeepSeek, Mixtral 等）。

## 4. 验证与追溯
*   **日志标识**: 启动时搜索 `[MoE]` 关键字。
*   **成功标志**: `[MoE] Rank 0: Kept X expert parameters, filtered Y.` 和 `[MoE] Successfully patched N MoE layers.`

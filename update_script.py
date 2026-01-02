import os

content = """import torch
import gc
import ray
import os
import time

class RayLatentPassAndKill:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "kill_and_pass"
    CATEGORY = "Raylight/Utils"

    def kill_and_pass(self, latent):
        print(">>> [RayKiller] 正在开始清理环境... <<<")
        
        # 1. 设置超时环境变量，帮助下次启动更快报错或恢复
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_TIMEOUT"] = "5000" # 5秒超时

        if ray.is_initialized():
            try:
                # 2. 强制杀掉残留 Actor
                # 如果有正在吊着的 RayTest 或 RayWorker，直接从管理层干掉
                from ray.util.state import list_actors
                actors = list_actors(filters=[("state", "==", "ALIVE")])
                for a in actors:
                    try:
                        handle = ray.get_actor(a['name'])
                        ray.kill(handle, no_reconstruction=True)
                    except:
                        pass
                
                print(f">>> [RayKiller] 已尝试强制清理 {len(actors)} 个 Actor <<<")
                
                ray.shutdown()
                print(">>> [RayKiller] Ray Cluster 已关闭 (Shutdown) <<<")
            except Exception as e:
                print(f">>> [RayKiller] 关闭 Ray 时发生错误: {e}")
        
        # 3. 显存大扫除
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # 4. 端口轮换 (这是核心：避开 300s TIME_WAIT 超时)
        current_port = int(os.environ.get("MASTER_PORT", "29500"))
        new_port = current_port + 1 if current_port < 31000 else 29500
        os.environ["MASTER_PORT"] = str(new_port)
        
        print(f">>> [RayKiller] 端口轮换: {current_port} -> {new_port} <<<")
        print(">>> [RayKiller] 环境重置完成。下一次启动将非常迅速。 <<<")

        return (latent,)

NODE_CLASS_MAPPINGS = {
    "RayLatentPassAndKill": RayLatentPassAndKill
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayLatentPassAndKill": "Ray Killer (Fixed Timeout)"
}
"""

target = r'\\192.168.2.126\Data\ComfyUI\custom_nodes\ray_killer.py'

try:
    with open(target, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated ray_killer.py")
except Exception as e:
    print(f"Error: {e}")

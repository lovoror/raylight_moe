import sys
from unittest.mock import MagicMock

# Comprehensive Mocking for Raylight Worker Test
mock_modules = [
    'ray', 'ray.exceptions', 'torch', 'torch.distributed', 'torch.distributed.tensor',
    'comfy', 'comfy.sd', 'comfy.sample', 'comfy.utils', 'comfy.patcher_extension',
    'raylight.distributed_modules.attention', 'raylight.distributed_modules.usp',
    'raylight.distributed_modules.cfg', 'raylight.comfy_dist.sd',
    'raylight.distributed_modules.moe', 'raylight.comfy_dist.model_management',
    'raylight.comfy_dist.model_patcher'
]

for mod in mock_modules:
    sys.modules[mod] = MagicMock()

import ray
import torch
from raylight.distributed_worker.ray_worker import RayWorker

def test_ray_worker_loading():
    # Setup worker
    # RayWorker.__init__ expects (local_rank, device_id, parallel_dict)
    parallel_dict = {"is_fsdp": False, "global_world_size": 2}
    worker = RayWorker(local_rank=0, device_id=0, parallel_dict=parallel_dict)
    
    # Mock load_torch_file
    mock_sd = {"test": "tensor_data"}
    import comfy.utils
    comfy.utils.load_torch_file.return_value = mock_sd
    
    # Test get_unet_state_dict
    ref = worker.get_unet_state_dict("dummy_path")
    comfy.utils.load_torch_file.assert_called_with("dummy_path")
    ray.put.assert_called_with(mock_sd)
    
    # Test load_unet_from_state_dict
    import comfy.sd
    worker.load_unet_from_state_dict(mock_sd, {"dtype": "fp16"})
    comfy.sd.load_diffusion_model_state_dict.assert_called_with(mock_sd, model_options={"dtype": "fp16"})
    assert worker.is_model_loaded == True
    
    print(">>> [Verification] RayWorker loading logic (Broadcast) verified successfully!")

if __name__ == "__main__":
    try:
        test_ray_worker_loading()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加Voyager utils到路径
VOYAGER_ROOT = Path(os.getenv("VOYAGER_ROOT", os.environ.get('MODEL_PATH', '') + '/HunyuanWorld-Voyager'))
sys.path.insert(0, str(VOYAGER_ROOT / 'voyager' / 'worldscore_utils'))

from camera_converter import CameraCoordinateConverter


def adapter_voyager(config, data, helper):
    """WorldScore → Voyager 适配器"""
    output_dir = data['output_dir']
    image_path = data['image_path']
    prompt_list = data['inpainting_prompt_list']
    cameras_ws = data['cameras']
    cameras_interp_ws = data['cameras_interp']
    camera_path = data['camera_path']
    anchor_frame_idx = data['anchor_frame_idx']
    num_scenes = data['num_scenes']

    helper.prepare_data(output_dir, data)

    resolution = config.get('resolution', [720, 1280])  # [H, W]
    height, width = resolution[0], resolution[1]

    from worldscore.benchmark.utils.utils import center_crop
    image_path = center_crop(image_path, (width, height), output_dir)

    print(f"\n{'='*60}")
    print(f"Voyager Adapter - Processing")
    print(f"{'='*60}")
    print(f"  Output dir: {output_dir}")
    print(f"  Num scenes: {num_scenes}")
    print(f"  Camera commands: {camera_path}")
    print(f"  Anchor frames: {anchor_frame_idx}")
    print(f"  Prompts received: {len(prompt_list)}")

    # 坐标系转换
    from camera_converter import CameraCoordinateConverter
    converter = CameraCoordinateConverter()

    cameras_voyager = [converter.worldscore_c2w_to_voyager_w2c(cam) for cam in cameras_ws]
    cameras_interp_voyager = [converter.worldscore_c2w_to_voyager_w2c(cam) for cam in cameras_interp_ws]

    # 生成Voyager内参
    focal_length = config.get('focal_length', 500)
    fx = fy = focal_length
    cx = width / 2
    cy = height / 2

    intrinsics = [np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                  for _ in range(len(cameras_interp_voyager))]

    # 保存相机数据
    voyager_data = {
        'cameras_voyager': [cam.tolist() for cam in cameras_voyager],
        'cameras_interp_voyager': [cam.tolist() for cam in cameras_interp_voyager],
        'intrinsics': [K.tolist() for K in intrinsics],
        'camera_path': camera_path,
        'anchor_frame_idx': anchor_frame_idx,
        'num_scenes': num_scenes,
        'resolution': [height, width],
        'focal_length': focal_length,
    }

    voyager_data_file = Path(output_dir) / 'voyager_camera_data.json'
    with open(voyager_data_file, 'w') as f:
        json.dump(voyager_data, f, indent=2)

    # ✓ 关键修改: I2V模型跳过第一个prompt (已经由input_image表示)
    if len(prompt_list) > 1:
        generation_prompts = prompt_list[1:]
        print(f"✓ I2V mode: Using {len(generation_prompts)} prompts (skipped first)")
    else:
        generation_prompts = prompt_list
        print(f"✓ Using {len(generation_prompts)} prompts")

    return image_path, generation_prompts

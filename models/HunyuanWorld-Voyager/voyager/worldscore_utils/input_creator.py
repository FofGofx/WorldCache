# WorldScore/models/HunyuanWorld-Voyager/voyager/worldscore_utils/input_creator.py

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch

# 添加Voyager路径
VOYAGER_ROOT = Path(os.getenv("VOYAGER_ROOT", str(Path(__file__).parent.parent.parent)))
sys.path.insert(0, str(VOYAGER_ROOT / "data_engine"))

# 导入Voyager的函数（需要修改create_input.py使其可导入）
from create_input import (
    render_from_cameras_videos,
    create_video_input,
    depth_to_world_coords_points
)

class VoyagerInputCreator:
    """
    为Voyager生成输入条件（基于自定义相机矩阵）
    """

    def __init__(
            self,
            moge_model,
            device="cuda"
    ):
        self.moge_model = moge_model
        self.device = device


    def create_input_from_cameras(
            self,
            image_path: str,
            cameras_w2c: List[np.ndarray],
            intrinsics: List[np.ndarray],
            output_dir: str,
            width: int = 512,
            height: int = 512
    ):
        """
        基于自定义相机矩阵生成Voyager输入条件
        """
        print(f"\n{'='*80}")
        print(f"[Tracker-INPUT_CREATOR] VoyagerInputCreator.create_input_from_cameras()")
        print(f"{'='*80}")
        print(f"  输入图像路径: {image_path}")
        print(f"  目标渲染分辨率 (W×H): {width} × {height}")
        print(f"  相机数量: {len(cameras_w2c)}")

        # 1. 读取图像
        image = np.array(Image.open(image_path).convert("RGB"))
        ori_height, ori_width = image.shape[:2]
        print(f"  读取的图像尺寸 (W×H): {ori_width} × {ori_height}")
        
        # 1.1 检查图像尺寸是否与目标分辨率匹配，如果不匹配则resize
        if ori_width != width or ori_height != height:
            print(f"  图像尺寸不匹配，需要resize: ({ori_width} × {ori_height}) → ({width} × {height})")
            # 使用PIL进行resize以保持图像质量
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            image = np.array(pil_image)
            print(f"  Resize完成，当前图像尺寸 (W×H): {width} × {height}")
        else:
            print(f"  图像尺寸已匹配，无需resize")

        # 2. 使用MoGe估计深度
        image_tensor = torch.tensor(
            image / 255,
            dtype=torch.float32,
            device=self.device
        ).permute(2, 0, 1)
        print(f"  传入MoGE的tensor形状 (C×H×W): {image_tensor.shape}")

        with torch.no_grad():
            output = self.moge_model.infer(image_tensor)

        depth = np.array(output['depth'].detach().cpu())
        depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4
        print(f"  MoGE输出深度图形状: {depth.shape}")

        # 3. Backproject点云
        point_map = depth_to_world_coords_points(
            depth, cameras_w2c[0], intrinsics[0]
        )
        points = point_map.reshape(-1, 3)
        colors = image.reshape(-1, 3)
        print(f"  生成点云数量: {points.shape[0]}")

        # 4. 渲染多视角 - 使用全分辨率
        print(f"  开始渲染多视角，目标分辨率 (W×H): {width} × {height}")
        render_list, mask_list, depth_list = render_from_cameras_videos(
            points, colors, cameras_w2c, intrinsics,
            height=height, width=width  # ✓ 修改：使用全分辨率
        )
        print(f"  渲染完成，共 {len(render_list)} 个视角")

        # 5. 创建video_input
        create_video_input(
            render_list, mask_list, depth_list, output_dir,
            separate=True, ref_image=image, ref_depth=depth,
            Width=width, Height=height
        )

        print(f"✓ Voyager input created at: {output_dir}")


def make_create_input_importable():
    """
    在create_input.py末尾添加：

    if __name__ == "__main__":
        # 原有的main代码

    这样就可以导入其中的函数
    """
    pass

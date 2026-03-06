# WorldCache/world_generators/aether.py

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from PIL import Image
import imageio

# 设置Aether路径
AETHER_ROOT = os.getenv("AETHER_ROOT", os.environ.get('MODEL_PATH', '') + '/Aether')
print(f"Aether root path set to: {AETHER_ROOT}")
sys.path.insert(0, AETHER_ROOT)

# 导入Aether相关模块
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from aether.pipelines.aetherv1_pipeline_cogvideox import AetherV1PipelineCogVideoX
from aether.utils.postprocess_utils import camera_pose_to_raymap

# 导入相机轨迹修正函数
from worldscore.benchmark.helpers.adapters.threedgen.adapter_aether import correct_camera_trajectory_for_aether


class Aether:
    """Aether模型包装器（用于WorldScore）"""

    def __init__(
            self,
            model_name: str,
            generation_type: str,
            aether_root: str,
            cogvideox_pretrained_model_name_or_path: str,
            aether_pretrained_model_name_or_path: str,
            resolution: List[int],
            frames: int = 41,
            fps: int = 12,
            num_inference_steps: int = 50,
            guidance_scale: float = 3.0,
            use_dynamic_cfg: bool = True,
            seed: int = 42,
            **kwargs
    ):
        self.model_name = model_name
        self.generation_type = generation_type
        self.aether_root = Path(aether_root)
        self.resolution = resolution  # [H, W]
        self.frames = frames
        self.fps = fps
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.use_dynamic_cfg = use_dynamic_cfg
        self.seed = seed

        # 设置PyTorch确定性参数以提高可重复性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 设置随机种子
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载pipeline
        self._load_pipeline(
            cogvideox_pretrained_model_name_or_path,
            aether_pretrained_model_name_or_path
        )

        # 当前工作目录（用于存储临时数据）
        self.current_output_dir = None

        print(f"✓ Aether initialized")
        print(f"  Root: {self.aether_root}")
        print(f"  Resolution: {self.resolution}")
        print(f"  Frames: {self.frames}, FPS: {self.fps}")

    def _load_pipeline(
            self,
            cogvideox_pretrained_model_name_or_path: str,
            aether_pretrained_model_name_or_path: str
    ):
        """加载Aether pipeline"""
        try:
            self.pipeline = AetherV1PipelineCogVideoX(
                tokenizer=AutoTokenizer.from_pretrained(
                    cogvideox_pretrained_model_name_or_path,
                    subfolder="tokenizer",
                ),
                text_encoder=T5EncoderModel.from_pretrained(
                    cogvideox_pretrained_model_name_or_path,
                    subfolder="text_encoder"
                ),
                vae=AutoencoderKLCogVideoX.from_pretrained(
                    cogvideox_pretrained_model_name_or_path,
                    subfolder="vae",
                    torch_dtype=torch.bfloat16,
                ),
                scheduler=CogVideoXDPMScheduler.from_pretrained(
                    cogvideox_pretrained_model_name_or_path,
                    subfolder="scheduler"
                ),
                transformer=CogVideoXTransformer3DModel.from_pretrained(
                    aether_pretrained_model_name_or_path,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                ),
            )
            self.pipeline.vae.enable_slicing()
            self.pipeline.vae.enable_tiling()
            self.pipeline.to(self.device)
            
            # Apply Worldcache acceleration
            from aether.worldcache_aether.forwards.apply_worldcache_aether_pipeline import apply_worldcache_aether_pipeline
            from aether.worldcache_aether.forwards.apply_worldcache_aether_forward import apply_worldcache_aether_forward
            
            apply_worldcache_aether_pipeline(self.pipeline)
            apply_worldcache_aether_forward(self.pipeline.transformer)
            
            print("✓ Aether pipeline loaded")
        except Exception as e:
            print(f"⚠ Failed to load Aether pipeline: {e}")
            raise

    def _load_camera_data(self, output_dir: Path):
        """从约定的位置加载相机数据"""
        camera_data_file = output_dir / 'aether_camera_data.json'

        if not camera_data_file.exists():
            print(f"⚠ Camera data file not found: {camera_data_file}")
            return None

        with open(camera_data_file, 'r') as f:
            data = json.load(f)

        # 加载raymap（从numpy文件）
        raymap_file = output_dir / data['raymap_file']
        if not raymap_file.exists():
            raise FileNotFoundError(f"Raymap file not found: {raymap_file}")
        raymap = np.load(raymap_file)

        # 转换回numpy数组
        return {
            'cameras_aether': [np.array(cam) for cam in data['cameras_aether']],
            'cameras_interp_aether': [np.array(cam) for cam in data['cameras_interp_aether']],
            'intrinsics': [np.array(K) for K in data['intrinsics']],
            'raymap': raymap,  # 从numpy文件加载
            'camera_path': data['camera_path'],
            'anchor_frame_idx': data['anchor_frame_idx'],
            'num_scenes': data['num_scenes'],
            'resolution': data['resolution'],
            'focal_length': data['focal_length'],
        }

    def generate_video(
            self,
            prompt: str,
            image_path: Optional[str] = None,
            **kwargs
    ) -> List[Image.Image]:
        """
        生成单个场景的视频

        注意：这个方法会被WorldScore框架多次调用（每个场景一次）

        Args:
            prompt: 当前场景的文本描述
            image_path: 输入图像路径
                - 第1次调用: "output_dir/input_image.png"
                - 第2次调用: "output_dir/input_image_1.png"
                - 第3次调用: "output_dir/input_image_2.png"

        Returns:
            frames: 生成的视频帧列表
        """
        if image_path is None:
            raise ValueError("Aether requires an input image (i2v model)")

        # 从image_path推断output_dir
        output_dir = Path(image_path).parent
        self.current_output_dir = output_dir

        # 加载相机数据
        camera_data = self._load_camera_data(output_dir)

        if camera_data is None:
            raise RuntimeError("Camera data not found. Please ensure adapter_aether has been called first.")

        # ✓ 关键修改: 通过image_path的文件名判断当前是第几个场景
        image_name = Path(image_path).name

        if image_name == "input_image.png":
            scene_idx = 0  # 第一个场景
        elif "_" in image_name:
            # 提取 input_image_N.png 中的 N
            try:
                scene_idx = int(image_name.split("_")[-1].split(".")[0])
            except ValueError:
                print(f"⚠ Failed to parse scene index from {image_name}, using 0")
                scene_idx = 0
        else:
            print(f"⚠ Unknown image name format: {image_name}, using scene_idx=0")
            scene_idx = 0

        print(f"\n{'='*60}")
        print(f"Aether.generate_video() - Scene {scene_idx + 1}/{camera_data['num_scenes']}")
        print(f"{'='*60}")
        print(f"  Image: {image_name}")
        print(f"  Prompt: {prompt}")
        print(f"  Scene index: {scene_idx}")

        return self._generate_single_scene_with_index(
            prompt, image_path, output_dir, camera_data, scene_idx
        )

    def _generate_single_scene_with_index(
            self,
            prompt: str,
            image_path: str,
            output_dir: Path,
            camera_data: dict,
            scene_idx: int
    ) -> List[Image.Image]:
        """
        根据scene_idx生成对应场景

        Args:
            prompt: 场景描述
            image_path: 输入图像路径
            output_dir: 输出目录
            camera_data: 相机数据字典
            scene_idx: 场景索引 (0, 1, 2, ...)

        Returns:
            frames: 生成的视频帧
        """
        anchor_frame_idx = camera_data['anchor_frame_idx']
        num_scenes = camera_data['num_scenes']
        full_raymap = camera_data['raymap']  # (N, 6, H//8, W//8)

        # 验证scene_idx
        if scene_idx < 0 or scene_idx >= num_scenes:
            print(f"⚠ Invalid scene_idx {scene_idx}, clamping to [0, {num_scenes-1}]")
            scene_idx = max(0, min(scene_idx, num_scenes - 1))

        # ✓ 正确提取该场景的相机轨迹
        start_idx = anchor_frame_idx[scene_idx]
        end_idx = anchor_frame_idx[scene_idx + 1]

        print(f"  Camera trajectory: frames [{start_idx}:{end_idx}] (total {end_idx - start_idx + 1} frames)")

        # 对于 scene_idx > 0 的场景，需要重新计算raymap使其相对于当前场景第一帧
        # 因为输入图像是上一个场景的最后一帧，但raymap应该是相对于当前场景第一帧的
        if scene_idx > 0:
            print(f"  Recomputing raymap relative to scene {scene_idx} first frame...")
            
            try:
                # 1. 提取当前场景的相机pose和内参
                cameras_interp_aether = camera_data['cameras_interp_aether']
                intrinsics = camera_data['intrinsics']
                camera_path = camera_data['camera_path']
                
                scene_cameras = cameras_interp_aether[start_idx:end_idx + 1]
                scene_intrinsics = intrinsics[start_idx:end_idx + 1]
                
                # 确保scene_cameras是numpy数组格式
                if isinstance(scene_cameras, list):
                    scene_cameras = np.stack(scene_cameras, axis=0)
                
                # 2. 获取当前场景的相机命令
                current_camera_command = None
                if isinstance(camera_path, list) and scene_idx < len(camera_path):
                    current_camera_command = camera_path[scene_idx]
                    print(f"  Current scene camera command: {current_camera_command}")
                
                # 3. 应用相机轨迹修正（如果需要）
                if current_camera_command:
                    print(f"  Applying trajectory correction for: {current_camera_command}")
                    scene_cameras_corrected = correct_camera_trajectory_for_aether(
                        scene_cameras.copy(),  # 使用副本避免修改原始数据
                        [current_camera_command]  # 传入当前场景的命令
                    )
                    # 如果返回的是列表，转换为numpy数组
                    if isinstance(scene_cameras_corrected, list):
                        scene_cameras_corrected = np.stack(scene_cameras_corrected, axis=0)
                    scene_cameras = scene_cameras_corrected
                
                # 4. 获取当前场景第一帧的pose作为新的参考原点
                # scene_cameras中的pose都是相对于整个序列第一帧的
                ref_pose = scene_cameras[0]  # 当前场景第一帧的pose（相对于整个序列第一帧）
                
                # 5. 验证ref_pose是否有效
                if np.any(np.isnan(ref_pose)) or np.any(np.isinf(ref_pose)):
                    print(f"  ⚠ Invalid ref_pose detected (NaN/Inf), using original raymap")
                    scene_raymap = full_raymap[start_idx:end_idx + 1]
                else:
                    # 6. 将所有pose转换为相对于当前场景第一帧
                    # relative_pose_i = inv(ref_pose) @ pose_i
                    # 这样新的relative_pose_i中，第一帧是单位矩阵
                    scene_cameras_relative = []
                    ref_pose_inv = np.linalg.inv(ref_pose)
                    
                    invalid_pose_detected = False
                    for i, pose in enumerate(scene_cameras):
                        relative_pose = ref_pose_inv @ pose
                        # 验证转换后的pose
                        if np.any(np.isnan(relative_pose)) or np.any(np.isinf(relative_pose)):
                            print(f"  ⚠ Invalid relative_pose detected at frame {i} (NaN/Inf), using original raymap")
                            invalid_pose_detected = True
                            break
                        scene_cameras_relative.append(relative_pose)
                    
                    if invalid_pose_detected:
                        scene_raymap = full_raymap[start_idx:end_idx + 1]
                    else:
                        # 7. 重新生成raymap（使用与适配器相同的参数）
                        camera_poses = np.stack(scene_cameras_relative, axis=0)
                        intrinsics_array = np.stack(scene_intrinsics, axis=0)
                        height, width = camera_data['resolution']
                        
                        scene_raymap = camera_pose_to_raymap(
                            camera_pose=camera_poses,
                            intrinsic=intrinsics_array,
                            ray_o_scale_factor=10.0,
                            dmax=1.0,
                            H=height,
                            W=width,
                            vae_downsample=8,
                            align_corners=False
                        )
                        
                        # 8. 验证生成的raymap
                        if np.any(np.isnan(scene_raymap)) or np.any(np.isinf(scene_raymap)):
                            print(f"  ⚠ Invalid raymap detected (NaN/Inf), using original raymap")
                            scene_raymap = full_raymap[start_idx:end_idx + 1]
                        else:
                            # 检查raymap的数值范围是否合理
                            raymap_min = np.min(scene_raymap)
                            raymap_max = np.max(scene_raymap)
                            if abs(raymap_min) > 1e6 or abs(raymap_max) > 1e6:
                                print(f"  ⚠ Raymap values out of reasonable range (min={raymap_min:.2e}, max={raymap_max:.2e}), using original raymap")
                                scene_raymap = full_raymap[start_idx:end_idx + 1]
                            else:
                                print(f"  ✓ Raymap recomputed: shape {scene_raymap.shape}")
            except Exception as e:
                print(f"  ⚠ Error during raymap recomputation: {e}")
                print(f"  Falling back to original raymap")
                import traceback
                traceback.print_exc()
                scene_raymap = full_raymap[start_idx:end_idx + 1]
        else:
            # scene_idx == 0，直接使用原有的raymap（已经相对于第一帧）
            scene_raymap = full_raymap[start_idx:end_idx + 1]  # (num_frames, 6, H//8, W//8)
        
        # 验证raymap的帧数
        num_frames_scene = scene_raymap.shape[0]
        if num_frames_scene != self.frames:
            print(f"⚠ Scene raymap has {num_frames_scene} frames, but expected {self.frames}")
            # 如果帧数不匹配，可能需要插值或截断
            if num_frames_scene > self.frames:
                scene_raymap = scene_raymap[:self.frames]
            else:
                # 重复最后一帧
                last_frame = scene_raymap[-1:]
                padding = np.repeat(last_frame, self.frames - num_frames_scene, axis=0)
                scene_raymap = np.concatenate([scene_raymap, padding], axis=0)

        # 创建场景子目录
        scene_dir = output_dir / f'scene_{scene_idx}'
        scene_dir.mkdir(exist_ok=True)

        # 加载输入图像
        image = Image.open(image_path).convert("RGB")

        # 运行Aether推理
        print(f"  Running Aether inference...")
        print(f"    Prompt: {prompt}")
        print(f"    Raymap shape: {scene_raymap.shape}")
        print(f"    Resolution: {self.resolution}")

        output = self.pipeline(
            task="prediction",
            image=image,
            raymap=scene_raymap,
            height=self.resolution[0],
            width=self.resolution[1],
            num_frames=self.frames,
            fps=self.fps,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            use_dynamic_cfg=self.use_dynamic_cfg,
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
            return_dict=True,
            output_dir=str(scene_dir),  # Pass scene directory for denoising_time.txt
        )

        # 提取RGB帧
        rgb_frames = output.rgb  # (num_frames, H, W, 3) in range [0, 1]
        
        # 转换为PIL Image列表
        frames = []
        for i in range(rgb_frames.shape[0]):
            frame = (rgb_frames[i] * 255).astype(np.uint8)
            frames.append(Image.fromarray(frame))

        print(f"  ✓ Generated {len(frames)} frames")

        # 保存场景视频（可选）
        video_output_path = scene_dir / 'output.mp4'
        self._save_video(frames, video_output_path)

        # ✓ 保存下一场景的输入图像（如果还有后续场景）
        next_scene_idx = scene_idx + 1
        if next_scene_idx < num_scenes:
            # 还有下一个场景,保存最后一帧作为输入
            last_frame_path = output_dir / f'input_image_{next_scene_idx}.png'
            frames[-1].save(last_frame_path)
            print(f"✓ Saved input for next scene: {last_frame_path}")
        else:
            print(f"✓ Last scene completed, no more inputs to save")

        return frames

    def _save_video(self, frames: List[Image.Image], output_path: Path):
        """保存视频文件"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # 使用imageio.v2或v3兼容的方式
            try:
                import imageio.v3 as iio
                iio.imwrite(
                    str(output_path),
                    np.stack([np.array(frame) for frame in frames]),
                    fps=self.fps,
                    codec='libx264',
                    pixelformat='yuv420p'
                )
            except ImportError:
                # 回退到旧版本
                imageio.mimwrite(
                    str(output_path),
                    [np.array(frame) for frame in frames],
                    fps=self.fps
                )
            print(f"  ✓ Video saved to: {output_path}")
        except Exception as e:
            print(f"⚠ Failed to save video: {e}")

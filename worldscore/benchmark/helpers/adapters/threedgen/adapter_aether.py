import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加Aether utils到路径
AETHER_ROOT = Path(os.getenv("AETHER_ROOT", os.environ.get('MODEL_PATH', '') + '/Aether'))
sys.path.insert(0, str(AETHER_ROOT))

from aether.utils.postprocess_utils import camera_pose_to_raymap


def correct_camera_trajectory_for_aether(cameras_interp_aether, camera_path):
    """
    修正 Aether 适配后的相机轨迹，使其符合论文示意图的要求。
    
    根据用户反馈：
    - Orbit left 被错误处理为 Pan left（只有旋转，没有位置变化）
    - Orbit right 被错误处理为 Move right 和 Pan right 中间地带
    - Pan left 被处理为了 Pan right（旋转方向相反）
    - Pan right 被处理为了 Pan left（旋转方向相反）
    
    修正策略：
    1. 对于 pan_left 和 pan_right：反转旋转方向（对旋转矩阵取转置）
    2. 对于 orbit_left 和 orbit_right：确保旋转方向正确，并确保有位置变化
    
    Args:
        cameras_interp_aether: List[np.ndarray] 或 np.ndarray, 形状为 (N, 4, 4) 的相机pose列表
        camera_path: List[str], 相机命令列表，如 ['orbit_left']
    
    Returns:
        corrected_cameras: 修正后的相机pose列表或数组
    """
    # 转换为numpy数组格式以便处理
    if isinstance(cameras_interp_aether, list):
        cameras_array = np.stack(cameras_interp_aether, axis=0)
        is_list = True
    else:
        cameras_array = cameras_interp_aether.copy()
        is_list = False
    
    # 获取相机命令类型（通常 camera_path 是列表，取第一个元素）
    if isinstance(camera_path, list) and len(camera_path) > 0:
        camera_command = camera_path[0]
    elif isinstance(camera_path, str):
        camera_command = camera_path
    else:
        print(f"  ⚠ Unknown camera_path format: {camera_path}, skipping correction")
        return cameras_interp_aether
    
    print(f"  Correcting trajectory for command: {camera_command}")
    
    # 需要修正的命令类型
    if camera_command in ['pan_left', 'pan_right', 'orbit_left', 'orbit_right']:
        num_frames = cameras_array.shape[0]
        R_first = cameras_array[0, :3, :3]  # 第一帧的旋转矩阵（应该是单位矩阵）
        T_first = cameras_array[0, :3, 3]  # 第一帧的平移（应该是零向量）
        
        if camera_command in ['pan_left', 'pan_right']:
            # Pan 操作：pan_left 和 pan_right 被完全交换了
            # pan_left 被处理为了 pan_right，pan_right 被处理为了 pan_left
            # 所以需要反转旋转方向来修正
            print(f"    Applying pan correction: reversing rotation direction (pan_left <-> pan_right swapped)")
            
            for i in range(1, num_frames):
                # 计算相对于第一帧的旋转
                R_current = cameras_array[i, :3, :3]
                R_rel = R_current @ R_first.T  # 相对旋转
                
                # 反转旋转方向：取转置（对于旋转矩阵，转置等于逆）
                # 这样 pan_left 的负角度会变成正角度，pan_right 的正角度会变成负角度
                R_rel_corrected = R_rel.T
                
                # 应用修正后的旋转（Pan 操作位置不变）
                cameras_array[i, :3, :3] = R_rel_corrected @ R_first
                # 保持平移不变（Pan 是原地旋转）
                cameras_array[i, :3, 3] = cameras_array[i, :3, 3]
        
        elif camera_command in ['orbit_left', 'orbit_right']:
            # Orbit 操作：修正旋转方向，并确保有位置变化
            print(f"    Applying orbit correction: fixing rotation direction and position")
            
            # 检测最后一帧的相对旋转和平移
            R_last = cameras_array[-1, :3, :3]
            T_last = cameras_array[-1, :3, 3]
            R_rel = R_last @ R_first.T
            
            # 提取旋转角度（绕Y轴）
            # 旋转矩阵绕Y轴旋转：R_y = [cos(θ), 0, sin(θ); 0, 1, 0; -sin(θ), 0, cos(θ)]
            angle = np.arctan2(R_rel[0, 2], R_rel[0, 0])
            angle_deg = np.degrees(angle)
            
            # 检查位置变化
            pos_change = np.linalg.norm(T_last - T_first)
            
            print(f"    Detected rotation angle: {angle_deg:.2f} degrees")
            print(f"    Detected position change: {pos_change:.4f}")
            
            # 根据命令类型判断期望的旋转方向
            need_reverse = False
            if camera_command == 'orbit_left':
                # Orbit left 应该是逆时针（正角度，从上方看）
                if angle < 0:
                    need_reverse = True
                    print(f"    Reversing orbit_left rotation direction")
            elif camera_command == 'orbit_right':
                # Orbit right 应该是顺时针（负角度，从上方看）
                if angle > 0:
                    need_reverse = True
                    print(f"    Reversing orbit_right rotation direction")
            
            # 如果位置变化太小，说明被错误处理为 Pan 操作
            if pos_change < 0.01:
                print(f"    ⚠ Warning: Orbit operation has minimal position change, may be incorrectly processed as Pan")
            
            # 应用旋转方向修正
            if need_reverse:
                for i in range(1, num_frames):
                    R_current = cameras_array[i, :3, :3]
                    T_current = cameras_array[i, :3, 3]
                    
                    # 计算相对旋转
                    R_rel = R_current @ R_first.T
                    # 反转旋转方向
                    R_rel_corrected = R_rel.T
                    
                    # 应用修正后的旋转
                    cameras_array[i, :3, :3] = R_rel_corrected @ R_first
                    # 保持平移不变（Orbit 的位置变化应该由 WorldScore 生成器正确生成）
    else:
        print(f"    No correction needed for command: {camera_command}")
    
    # 转换回原始格式
    if is_list:
        return [cameras_array[i] for i in range(cameras_array.shape[0])]
    else:
        return cameras_array


def adapter_aether(config, data, helper):
    """WorldScore → Aether 适配器"""
    output_dir = data['output_dir']
    image_path = data['image_path']
    prompt_list = data['inpainting_prompt_list']
    cameras_ws = data['cameras']  # WorldScore c2w格式
    cameras_interp_ws = data['cameras_interp']  # WorldScore c2w格式
    camera_path = data['camera_path']
    anchor_frame_idx = data['anchor_frame_idx']
    num_scenes = data['num_scenes']

    helper.prepare_data(output_dir, data)

    resolution = config.get('resolution', [480, 720])  # [H, W]
    height, width = resolution[0], resolution[1]

    from worldscore.benchmark.utils.utils import center_crop
    image_path = center_crop(image_path, (width, height), output_dir)

    print(f"\n{'='*60}")
    print(f"Aether Adapter - Processing")
    print(f"{'='*60}")
    print(f"  Output dir: {output_dir}")
    print(f"  Num scenes: {num_scenes}")
    print(f"  Camera commands: {camera_path}")
    print(f"  Anchor frames: {anchor_frame_idx}")
    print(f"  Prompts received: {len(prompt_list)}")

    # WorldScore使用Blender右手坐标系（Y-up），c2w格式
    # Aether要求相机pose相对于第一帧的相机坐标系（见README）
    # 需要将所有pose转换到第一帧的相机坐标系
    cameras_interp_ws = [np.array(cam, dtype=np.float32) for cam in cameras_interp_ws]

    # 获取第一帧的c2w矩阵
    c2w_first = cameras_interp_ws[0]  # (4, 4)
    # 转换为w2c（world-to-camera）
    w2c_first = np.linalg.inv(c2w_first)

    # 将所有相机pose转换到第一帧的相机坐标系
    # relative_pose_i = w2c_first @ c2w_i
    # 这样第一帧的relative_pose是单位矩阵，后续帧都是相对于第一帧的
    cameras_aether = []
    cameras_interp_aether = []

    for cam in cameras_ws:
        cam_array = np.array(cam, dtype=np.float32)
        relative_pose = w2c_first @ cam_array
        # Aether的Z轴方向与WorldScore相反，需要对Z轴平移分量取反
        relative_pose[2, 3] = -relative_pose[2, 3]
        cameras_aether.append(relative_pose)

    for cam in cameras_interp_ws:
        relative_pose = w2c_first @ cam
        # Aether的Z轴方向与WorldScore相反，需要对Z轴平移分量取反
        relative_pose[2, 3] = -relative_pose[2, 3]
        cameras_interp_aether.append(relative_pose)

    print(f"  ✓ Converted cameras to first frame's coordinate system")
    print(f"    First frame pose (should be identity):")
    print(f"      Translation: {cameras_interp_aether[0][:3, 3]}")
    print(f"      Rotation matrix is identity: {np.allclose(cameras_interp_aether[0][:3, :3], np.eye(3), atol=1e-5)}")

    # 生成内参矩阵
    focal_length = config.get('focal_length', 500)
    fx = fy = focal_length
    cx = width / 2
    cy = height / 2

    intrinsics = [np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                  for _ in range(len(cameras_interp_aether))]

    # 将相机轨迹转换为raymap格式
    print(f"  Converting camera trajectories to raymap format...")
    print(f"    Total frames: {len(cameras_interp_aether)}")
    print(f"    Resolution: {height}x{width}")
    
    # 修正相机轨迹（根据 camera_path 修正特定轨迹类型的旋转方向）
    print(f"  Applying trajectory corrections for Aether...")
    cameras_interp_aether = correct_camera_trajectory_for_aether(cameras_interp_aether, camera_path)
    
    # 准备相机pose数组 (N, 4, 4)
    camera_poses = np.stack(cameras_interp_aether, axis=0)
    intrinsics_array = np.stack(intrinsics, axis=0)
    
    # 调用camera_pose_to_raymap转换
    # 注意：使用默认参数 ray_o_scale_factor=10.0, dmax=1.0
    raymap = camera_pose_to_raymap(
        camera_pose=camera_poses,
        intrinsic=intrinsics_array,
        ray_o_scale_factor=10.0,
        dmax=1.0,
        H=height,
        W=width,
        vae_downsample=8,
        align_corners=False
    )
    
    print(f"  ✓ Raymap generated: shape {raymap.shape}")

    # 保存raymap为numpy文件（因为raymap可能很大）
    raymap_file = Path(output_dir) / 'aether_raymap.npy'
    np.save(raymap_file, raymap)
    print(f"  ✓ Raymap saved to: {raymap_file}")

    # 保存相机数据和元数据（不包含raymap，只保存路径）
    aether_data = {
        'cameras_aether': [cam.tolist() for cam in cameras_aether],
        'cameras_interp_aether': [cam.tolist() for cam in cameras_interp_aether],
        'intrinsics': [K.tolist() for K in intrinsics],
        'raymap_file': str(raymap_file.name),  # 只保存文件名
        'camera_path': camera_path,
        'anchor_frame_idx': anchor_frame_idx,
        'num_scenes': num_scenes,
        'resolution': [height, width],
        'focal_length': focal_length,
        'raymap_shape': list(raymap.shape),  # 保存shape信息
    }

    aether_data_file = Path(output_dir) / 'aether_camera_data.json'
    with open(aether_data_file, 'w') as f:
        json.dump(aether_data, f, indent=2)
    
    print(f"  ✓ Camera data saved to: {aether_data_file}")

    # ✓ 关键修改: I2V模型跳过第一个prompt (已经由input_image表示)
    if len(prompt_list) > 1:
        generation_prompts = prompt_list[1:]
        print(f"✓ I2V mode: Using {len(generation_prompts)} prompts (skipped first)")
    else:
        generation_prompts = prompt_list
        print(f"✓ Using {len(generation_prompts)} prompts")

    return image_path, generation_prompts

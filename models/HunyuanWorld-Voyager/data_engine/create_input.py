#WorldScore/models/HunyuanWorld-Voyager/data_engine/create_input.py
import numpy as np
from PIL import Image
import torch
import argparse
import os
import json
import imageio
import pyexr
import cv2
import os
import sys
from pathlib import Path

# 添加 MoGe 目录到 Python 路径
VOYAGER_ROOT = Path(os.getenv("VOYAGER_ROOT", str(Path(__file__).parent.parent)))
print(f"VOYAGER_ROOT: {VOYAGER_ROOT}")

MOGE_ROOT = VOYAGER_ROOT / "MoGe"
if str(MOGE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOGE_ROOT))

try:
    from moge.model.v1 import MoGeModel
except ImportError as e:
    print(f"Failed to load MoGE: {e}")
    # 如果导入失败，尝试从本地路径导入
    try:
        # 确保 MoGe 目录存在
        if not MOGE_ROOT.exists():
            raise RuntimeError(f"MoGe directory not found at {MOGE_ROOT}")
        from moge.model.v1 import MoGeModel
    except ImportError:
        raise RuntimeError("MoGE model not loaded: Failed to import MoGeModel. Please ensure MoGe is properly installed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/home/vllm/fgx/WorldScore/models/HunyuanWorld-Voyager/examples/case12/000.jpg")
    parser.add_argument("--render_output_dir", type=str, default="/home/vllm/fgx/WorldScore/models/HunyuanWorld-Voyager/examples/case12/")
    parser.add_argument("--type", type=str, default="forward",
        choices=["forward", "backward", "left", "right", "turn_left", "turn_right"])
    # 新增参数：支持自定义相机
    parser.add_argument("--custom_cameras", type=str, default=None,
                        help="Path to custom camera JSON file (w2c format)")
    parser.add_argument("--custom_intrinsics", type=str, default=None,
                        help="Path to custom intrinsics JSON file")
    return parser.parse_args()


def camera_list(
        num_frames=49,
        type="forward",
        Width=512,
        Height=512,
        fx=256,
        fy=256,
        custom_cameras_w2c=None,  # 新增参数
        custom_intrinsics=None    # 新增参数
):
    """
    生成相机列表

    新增功能：支持自定义相机矩阵
    """
    if custom_cameras_w2c is not None and custom_intrinsics is not None:
        # 使用自定义相机
        print(f"Using custom cameras: {len(custom_cameras_w2c)} frames")

        extrinsics = np.array(custom_cameras_w2c)
        intrinsics = np.array(custom_intrinsics)

        return intrinsics, extrinsics

    # 原有的默认相机生成逻辑
    assert type in ["forward", "backward", "left", "right", "turn_left", "turn_right"], \
        "Invalid camera type"

    start_pos = np.array([0, 0, 0])
    end_pos = np.array([0, 0, 0])

    if type == "forward":
        end_pos = np.array([0, 0, 1])
    elif type == "backward":
        end_pos = np.array([0, 0, -1])
    elif type == "left":
        end_pos = np.array([-1, 0, 0])
    elif type == "right":
        end_pos = np.array([1, 0, 0])

    cx = Width // 2
    cy = Height // 2

    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    intrinsics = np.stack([intrinsic] * num_frames)

    # 插值相机位置
    camera_centers = np.linspace(start_pos, end_pos, num_frames)
    target_start = np.array([0, 0, 100])

    if type == "turn_left":
        target_end = np.array([-100, 0, 0])
    elif type == "turn_right":
        target_end = np.array([100, 0, 0])
    else:
        target_end = np.array([0, 0, 100])

    target_points = np.linspace(target_start, target_end, num_frames * 2)[:num_frames]

    extrinsics = []
    for t, target_point in zip(camera_centers, target_points):
        if type == "left" or type == "right":
            target_point = t + target_point

        z = (target_point - t)
        z = z / np.linalg.norm(z)
        x = np.array([1, 0, 0])
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)

        R = np.stack([x, y, z], axis=0)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ t
        extrinsics.append(w2c)

    extrinsics = np.stack(extrinsics)

    return intrinsics, extrinsics


# from VGGT: https://github.com/facebookresearch/vggt/blob/main/vggt/utils/geometry.py
def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4).

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points



def render_from_cameras_videos(points, colors, extrinsics, intrinsics, height, width):
    print(f"\n{'='*80}")
    print(f"[Tracker-RENDER] render_from_cameras_videos()")
    print(f"{'='*80}")
    print(f"  点云数量: {points.shape[0]}")
    print(f"  目标渲染尺寸 (W×H): {width} × {height}")
    print(f"  相机数量: {len(extrinsics)}")

    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

    render_list = []
    mask_list = []
    depth_list = []

    for frame_idx in range(len(extrinsics)):
        if frame_idx == 0:
            print(f"\n  处理第 {frame_idx} 帧...")

        extrinsic = extrinsics[frame_idx]
        intrinsic = intrinsics[frame_idx]

        camera_coords = (extrinsic @ homogeneous_points.T).T[:, :3]
        projected = (intrinsic @ camera_coords.T).T
        uv = projected[:, :2] / projected[:, 2].reshape(-1, 1)
        depths = projected[:, 2]

        pixel_coords = np.round(uv).astype(int)
        valid_pixels = (
                (pixel_coords[:, 0] >= 0) &
                (pixel_coords[:, 0] < width) &
                (pixel_coords[:, 1] >= 0) &
                (pixel_coords[:, 1] < height)
        )

        pixel_coords_valid = pixel_coords[valid_pixels]
        colors_valid = colors[valid_pixels]
        depths_valid = depths[valid_pixels]

        valid_mask = (depths_valid > 0) & (depths_valid < 60000)
        colors_valid = colors_valid[valid_mask]
        depths_valid = depths_valid[valid_mask]
        pixel_coords_valid = pixel_coords_valid[valid_mask]

        depth_buffer = np.full((height, width), np.inf)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        if len(pixel_coords_valid) > 0:
            rows = pixel_coords_valid[:, 1]
            cols = pixel_coords_valid[:, 0]

            sorted_idx = np.argsort(depths_valid)
            rows = rows[sorted_idx]
            cols = cols[sorted_idx]
            depths_sorted = depths_valid[sorted_idx]
            colors_sorted = colors_valid[sorted_idx]

            depth_buffer[rows, cols] = np.minimum(
                depth_buffer[rows, cols],
                depths_sorted
            )

            flat_indices = rows * width + cols
            unique_indices, idx = np.unique(flat_indices, return_index=True)

            final_rows = unique_indices // width
            final_cols = unique_indices % width

            image[final_rows, final_cols] = colors_sorted[idx, :3].astype(np.uint8)

            if frame_idx == 0:
                print(f"    有效投影点数: {len(pixel_coords_valid)}")
                print(f"    渲染后图像形状: {image.shape}")

        mask = np.zeros_like(depth_buffer, dtype=np.uint8)
        mask[depth_buffer != np.inf] = 255

        render_list.append(image)
        mask_list.append(mask)
        depth_list.append(depth_buffer)

    print(f"  渲染完成，共 {len(render_list)} 帧")
    return render_list, mask_list, depth_list


def create_video_input(
        render_list, mask_list, depth_list, render_output_dir,
        separate=True, ref_image=None, ref_depth=None,
        Width=512, Height=512,
        min_percentile=2, max_percentile=98
):
    print(f"\n{'='*80}")
    print(f"[Tracker-VIDEO_INPUT] create_video_input()")
    print(f"{'='*80}")
    print(f"  输出目录: {render_output_dir}")
    print(f"  目标尺寸 (W×H): {Width} × {Height}")
    print(f"  输入帧数: {len(render_list)}")

    video_output_dir = os.path.join(render_output_dir)
    os.makedirs(video_output_dir, exist_ok=True)
    video_input_dir = os.path.join(render_output_dir, "video_input")
    os.makedirs(video_input_dir, exist_ok=True)

    value_list = []
    for i, (render, mask, depth) in enumerate(zip(render_list, mask_list, depth_list)):
        if i == 0:
            print(f"\n  处理第 {i} 帧...")
            print(f"    渲染图尺寸 (H×W×C): {render.shape}")
            print(f"    深度图尺寸: {depth.shape}")

        mask = mask > 0
        depth[mask] = 1 / (depth[mask] + 1e-6)
        depth_values = depth[mask]

        min_percentile = np.percentile(depth_values, 2)
        max_percentile = np.percentile(depth_values, 98)
        value_list.append((min_percentile, max_percentile))

        depth[mask] = (depth[mask] - min_percentile) / (max_percentile - min_percentile)
        depth[~mask] = depth[mask].min()

        # ✓ 修改：只在尺寸不匹配时才resize
        if render.shape[1] != Width or render.shape[0] != Height:
            print(f"    需要缩放：{render.shape[:2]} → ({Height}, {Width})")
            render = cv2.resize(render, (Width, Height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize((mask.astype(np.float32) * 255).astype(np.uint8), \
                              (Width, Height), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (Width, Height), interpolation=cv2.INTER_LINEAR)
        else:
            mask = (mask.astype(np.float32) * 255).astype(np.uint8)

        if i == 0:
            print(f"    最终渲染图尺寸 (H×W×C): {render.shape}")
            print(f"    最终深度图尺寸: {depth.shape}")

        # Save mask as png
        mask_path = os.path.join(video_input_dir, f"mask_{i:04d}.png")
        imageio.imwrite(mask_path, mask)

        if separate:
            render_path = os.path.join(video_input_dir, f"render_{i:04d}.png")
            imageio.imwrite(render_path, render)
            depth_path = os.path.join(video_input_dir, f"depth_{i:04d}.exr")
            pyexr.write(depth_path, depth)

            if i == 0:
                print(f"    保存渲染图到: {render_path}")
                print(f"    保存深度图到: {depth_path}")

        if i == 0:
            if separate:
                ref_image_path = os.path.join(video_output_dir, f"ref_image.png")
                imageio.imwrite(ref_image_path, ref_image)
                ref_depth_path = os.path.join(video_output_dir, f"ref_depth.exr")
                pyexr.write(ref_depth_path, depth)
                print(f"  保存参考图到: {ref_image_path}")

    with open(os.path.join(video_output_dir, f"depth_range.json"), "w") as f:
        json.dump(value_list, f)

    print(f"  创建video_input完成")


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda")
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl", local_files_only=True).to(device)

    image = np.array(Image.open(args.image_path).convert("RGB").resize((1280, 720)))
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    output = model.infer(image_tensor)
    depth = np.array(output['depth'].detach().cpu())
    depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4

    Height, Width = image.shape[:2]

    # 加载自定义相机（如果提供）
    custom_cameras_w2c = None
    custom_intrinsics = None

    if args.custom_cameras and os.path.exists(args.custom_cameras):
        with open(args.custom_cameras, 'r') as f:
            custom_cameras_w2c = json.load(f)
        print(f"Loaded custom cameras from: {args.custom_cameras}")

    if args.custom_intrinsics and os.path.exists(args.custom_intrinsics):
        with open(args.custom_intrinsics, 'r') as f:
            custom_intrinsics = json.load(f)
        print(f"Loaded custom intrinsics from: {args.custom_intrinsics}")

    # 生成相机
    intrinsics, extrinsics = camera_list(
        num_frames=1,
        type=args.type,
        Width=Width,
        Height=Height,
        fx=256,
        fy=256,
        custom_cameras_w2c=custom_cameras_w2c,
        custom_intrinsics=custom_intrinsics
    )

    # Backproject点云
    point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
    points = point_map.reshape(-1, 3)
    colors = image.reshape(-1, 3)

    # 渲染多视角（使用49帧）
    intrinsics, extrinsics = camera_list(
        num_frames=49,
        type=args.type,
        Width=Width//2,
        Height=Height//2,
        fx=128,
        fy=128,
        custom_cameras_w2c=custom_cameras_w2c,
        custom_intrinsics=custom_intrinsics
    )

    render_list, mask_list, depth_list = render_from_cameras_videos(
        points, colors, extrinsics, intrinsics, height=Height//2, width=Width//2
    )

    create_video_input(
        render_list, mask_list, depth_list, args.render_output_dir, separate=True,
        ref_image=image, ref_depth=depth, Width=Width, Height=Height
    )


def main(image_dir, output_dir):
    print(f"\n{'='*80}")
    print(f"[Tracker-CREATE_INPUT] Voyager create_input.py - main()")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    model.eval()

    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    image_paths = sorted(itertools.chain(*(Path(image_dir).rglob(f'*.{suffix}') for suffix in include_suffices)))

    # 检查输出目录中已有的EXR文件数量
    output_exr_files = list(Path(output_dir).glob('*.exr'))
    if len(output_exr_files) >= len(image_paths):
        return

    for image_path in image_paths:
        print(f"\n[Tracker-5] 处理图像: {image_path}")

        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        print(f"  读取后图像形状 (H×W×C): {image.shape}")

        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        print(f"  转换为tensor后形状 (C×H×W): {image_tensor.shape}")

        # Inference
        output = model.infer(image_tensor, fov_x=None, resolution_level=9, num_tokens=None, use_fp16=True)
        depth = output['depth'].cpu().numpy()
        print(f"  MoGE生成的深度图形状: {depth.shape}")

        exr_output_dir = Path(output_dir)
        exr_output_dir.mkdir(exist_ok=True, parents=True)

        filename = f"{image_path.stem}.exr"
        save_file = exr_output_dir.joinpath(filename)

        cv2.imwrite(str(save_file), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
        print(f"  深度图保存到: {save_file}")

#WorldScore/data_engine/metric3d_infer.py
import os
import cv2
import argparse
import torch
import itertools
import json
from pathlib import Path
from typing import *
import pyexr

def main(image_dir, intrinsic_path, output_dir):    
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    image_paths = sorted(itertools.chain(*(Path(image_dir).rglob(f'*.{suffix}') for suffix in include_suffices)))

    # load model
    model = torch.hub.load("Metric3D", 'metric3d_vit_giant2', pretrain=True, source='local')
    model = model.to(device)
    model.eval()

    with open(intrinsic_path, 'r') as f:
        colmap_data = json.load(f)

    # Sort JSON keys by frame number (001, 002...109)
    sorted_frame_ids = sorted(colmap_data.keys(), key=lambda x: int(x))
    # Generate intrinsic list in order
    intrinsic_list = [colmap_data[frame_id]['intrinsic'] for frame_id in sorted_frame_ids]

    if len(image_paths) != len(intrinsic_list):
        raise ValueError(f"Number of images ({len(image_paths)}) does not match JSON frames ({len(intrinsic_list)})")

    # Check existing EXR files in output directory
    output_exr_files = list(Path(output_dir).glob('*.exr'))
    if len(output_exr_files) >= len(image_paths):
        return

    for idx, image_path in enumerate(image_paths):
        # Get corresponding intrinsic data by index
        intrinsic_data = intrinsic_list[idx]
        fx = intrinsic_data[0][0]
        fy = intrinsic_data[1][1]
        cx = intrinsic_data[0][2]
        cy = intrinsic_data[1][2]
        intrinsic = [fx, fy, cx, cy]

        # print(f"Processing image {image_path}")
                
        # load image
        rgb_origin = cv2.imread(str(image_path))[:, :, ::-1]

        # Adjust input size to fit pretrained model
        input_size = (616, 1064) # for vit model
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        # Remember to scale intrinsic, hold depth
        intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
        # Padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, \
            pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        # Normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()

        # Canonical camera space
        # inference
        with torch.no_grad():
            pred_depth, _, _ = model.inference({'input': rgb})

        # Unpad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], \
            pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        
        # Upsample to original size
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], \
            rgb_origin.shape[:2], mode='bilinear').squeeze()
        
        # Canonical camera space

        # De-canonical transform
        canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric

        depth = pred_depth.cpu().numpy()

        exr_output_dir = Path(output_dir)
        exr_output_dir.mkdir(exist_ok=True, parents=True)

        # Construct filename (use image_path stem directly)
        filename = f"{image_path.stem}.exr"
        save_file = exr_output_dir.joinpath(filename)  
        pyexr.write(save_file, depth[..., None], channel_names=["Y"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run metric3d data engine.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to input images directory')
    parser.add_argument('--intrinsic_path', type=str, required=True, help='Path to intrinsic file')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    main(args.image_dir, args.intrinsic_path, args.output_dir)

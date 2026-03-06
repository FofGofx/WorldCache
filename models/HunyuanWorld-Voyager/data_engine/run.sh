#!/bin/bash

IMAGE_DIR="your_input_path"
OUTPUT_DIR="your_output_path"
CUDA_DEVICE=0  

# 1. run vggt.py
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 vggt_infer.py --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR/vggt"

# # 2. run moge.py
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 moge_infer.py --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR/moge"

# # 3. run metric3d.py
INTRINSIC_PATH="$OUTPUT_DIR/vggt/colmap_data.json"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 metric3d_infer.py --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR/metric3d" --intrinsic_path "$INTRINSIC_PATH"

# # 4. conduct depth alignment
MOGE_DEPTH_DIR="$OUTPUT_DIR/moge"  
VGGT_DEPTH_DIR="$OUTPUT_DIR/vggt"  
METRIC3D_DEPTH_DIR="$OUTPUT_DIR/metric3d"  
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 depth_align.py \
    --image_dir "$IMAGE_DIR" \
    --moge_depth_dir "$MOGE_DEPTH_DIR" \
    --vggt_depth_dir "$VGGT_DEPTH_DIR/depth" \
    --metric3d_depth_dir "$METRIC3D_DEPTH_DIR" \
    --vggt_camera_json_file "$OUTPUT_DIR/vggt/colmap_data.json" \
    --output_dir "$OUTPUT_DIR/final"

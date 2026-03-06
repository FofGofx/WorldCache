#app.py
import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import gradio as gr
import imageio
import numpy as np
import torch
from PIL import Image
import uuid

from voyager.utils.file_utils import save_videos_grid
from voyager.config import parse_args
from voyager.inference import HunyuanVideoSampler

from moge.model.v1 import MoGeModel
from data_engine.create_input import camera_list, depth_to_world_coords_points, render_from_cameras_videos, create_video_input


def load_models(args):
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    model = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)

    return model


def generate_video(temp_path, prompt):
    condition_path = temp_path
    output_path = os.path.join(os.path.dirname(condition_path), "output")
    os.system(f"ALLOW_RESIZE_FOR_SP=1 torchrun --nproc_per_node=8 \
        sample_image2video.py \
        --model HYVideo-T/2 \
        --input-path \"{condition_path}\" \
        --prompt \"{prompt}\" \
        --i2v-stability \
        --infer-steps 50 \
        --flow-reverse \
        --flow-shift 7.0 \
        --seed 0 \
        --embedded-cfg-scale 6.0 \
        --save-path {output_path} \
        --ulysses-degree 8 \
        --ring-degree 1"
    )
    video_name = os.listdir(output_path)[0]
    return os.path.join(output_path, video_name)


def create_condition(model, image_path, direction, save_path):
    image = np.array(Image.open(image_path).resize((1280, 720)))
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device="cuda:0").permute(2, 0, 1)    
    output = model.infer(image_tensor)
    depth = np.array(output['depth'].detach().cpu())
    depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4
    
    Height, Width = image.shape[:2]

    intrinsics, extrinsics = camera_list(
        num_frames=1, type=direction, Width=Width, Height=Height, fx=256, fy=256
    )

    # Backproject point cloud
    point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
    points = point_map.reshape(-1, 3)
    colors = image.reshape(-1, 3)
    
    intrinsics, extrinsics = camera_list(
        num_frames=49, type=direction, Width=Width//2, Height=Height//2, fx=128, fy=128
    )
    render_list, mask_list, depth_list = render_from_cameras_videos(
        points, colors, extrinsics, intrinsics, height=Height//2, width=Width//2
    )
    
    create_video_input(
        render_list, mask_list, depth_list, os.path.join(save_path, "condition"), separate=True, 
        ref_image=image, ref_depth=depth, Width=Width, Height=Height)

    image_list = []
    for i in range(49):
        image_list.append(np.array(Image.open(os.path.join(save_path, "condition/video_input", f"render_{i:04d}.png"))))
    imageio.mimsave(os.path.join(save_path, "condition.mp4"), image_list, fps=8)
        
    return os.path.join(save_path, "condition.mp4")


def save_uploaded_image(image, save_dir="temp_images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    if image is None:
        return None
    
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    image_path = os.path.join(save_dir, "input_image.png")
    pil_image.save(image_path)
    return image_path


def create_video_demo():
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to("cuda:0")

    def process_condition_generation(image, direction):
        temp_path = os.path.join("temp", uuid.uuid4().hex[:8])
        image_path = save_uploaded_image(image, temp_path)
        assert image_path is not None, "Please upload image"
        condition_video_path = create_condition(moge_model, image_path, direction, temp_path)
        return os.path.join(temp_path, "condition"), condition_video_path
    
    def process_video_generation(temp_path, prompt):
        if temp_path is None or prompt is None:
            return None
        
        final_video_path = generate_video(temp_path, prompt)
        
        return final_video_path
    
    with gr.Blocks(title="Voyager Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ☯️ HunyuanWorld-Voyager")
        gr.Markdown("Upload an image, input description text, select movement direction, and generate exciting videos!")

        temp_path = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300
                )
                
                direction_choice = gr.Dropdown(
                    choices=[
                        "forward", "backward", "left", "right"
                    ],
                    label="Choose Camera Movement",
                    value="forward"
                )

                condition_video_output = gr.Video(
                    label="Condition Video",
                    height=300
                )
                
                condition_btn = gr.Button(
                    "⚙️ Generate Condition",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                input_prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Please input video description",
                    lines=3
                )
                
                gr.Markdown("### 🎥 Generating Final Video")
                final_video_output = gr.Video(
                    label="Generated Video", 
                    height=600
                )

                generate_btn = gr.Button(
                    "🚀 Generate Video",
                    variant="primary",
                    size="lg"
                )
        
        examples = []
        for i in range(1, 11):
            items = [os.path.join("examples", f"case{i}", "ref_image.png"), os.path.join("examples", f"case{i}", "condition.mp4")]
            with open(os.path.join("examples", f"case{i}", "prompt.txt"), "r") as f:
                prompt = f.readline()[:-1]
                items.append(prompt)
            items.append(os.path.join("examples", f"case{i}"))
            examples.append(items)

        def update_state(hidden_input):
            return str(hidden_input)

        hidden_input = gr.Textbox(visible=False)
            
        gr.Examples(
            examples=examples,
            inputs=[input_image, condition_video_output, input_prompt, hidden_input],
            outputs=[temp_path]
        )

        hidden_input.change(fn=update_state, inputs=hidden_input, outputs=temp_path)
        
        condition_btn.click(
            fn=process_condition_generation,
            inputs=[input_image, direction_choice],
            outputs=[temp_path, condition_video_output]
        )
        generate_btn.click(
            fn=process_video_generation,
            inputs=[temp_path, input_prompt],
            outputs=[final_video_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_video_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=True,
        debug=True
    )

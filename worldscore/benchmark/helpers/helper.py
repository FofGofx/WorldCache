#WorldScore/worldscore/benchmark/helpers/helper.py
import os
from PIL import Image
from torchvision.transforms import ToPILImage
import torch
from worldscore.benchmark.utils.utils import save_frames, merge_video
from worldscore.benchmark.utils.get_utils import get_adapter
import json
import time

class Helper():
    def __init__(self, config):
        self.focal_length = config.get('focal_length', 500)
        self.path = None
        self.config = config
        self.adapter = get_adapter(config)
        self.data = None
        
        self.frames = config['frames']
        self.num_scenes = None
        
        self.start_time = None
        self.end_time = None
        self.total_time = None
        
    def set_path(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.path = output_dir
    
    def store_data(self, data):
        visual_movement = data['image_data']['visual_movement']
        self.data = data['image_data']
        self.data['num_scenes'] = data['num_scenes']
        self.data['total_frames'] = data['total_frames']
        if visual_movement == "static":
            self.data['anchor_frame_idx'] = data['anchor_frame_idx']
        with open(f"{self.path}/image_data.json", 'w') as f:
            json.dump(self.data, f, indent=4)
    
    def prepare_data(self, output_dir, data):
        self.set_path(output_dir)
        self.store_data(data)
        self.num_scenes = data['num_scenes']
    
    def adapt(self, data):
        self.start_time = time.time()
        return self.adapter(self.config, data, self)
    
    def save_image(self, last_frame, image_path, i):
        # Get the directory and extension from the path
        directory = os.path.dirname(image_path)
        ext = os.path.splitext(image_path)[1]
        # Create new path with input_image_{i} format
        image_path = os.path.join(directory, f"input_image_{i}{ext}")
                    
        if isinstance(last_frame, Image.Image):
            last_frame.save(image_path)
        elif isinstance(last_frame, torch.Tensor): # [3, h, w] (0, 1)
            last_frame = ToPILImage()(last_frame)
            last_frame.save(image_path)
        return image_path
            
    def save(self, all_interpframes):
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        # Try to read denoising time from file(s)
        # For multi-scene generation, denoising_time.txt may be in scene subdirectories
        denoising_time = None
        denoising_time_file = os.path.join(self.path, 'denoising_time.txt')
        
        # First, try to read from main output directory
        if os.path.exists(denoising_time_file):
            try:
                with open(denoising_time_file, 'r') as f:
                    denoising_time = float(f.read().strip())
            except Exception as e:
                print(f"Warning: Failed to read denoising_time.txt from {denoising_time_file}: {e}")
        
        # If not found, try to find and accumulate from scene subdirectories
        if denoising_time is None:
            scene_dirs = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d)) and d.startswith('scene_')]
            if scene_dirs:
                total_denoising_time = 0.0
                found_count = 0
                for scene_dir in sorted(scene_dirs):
                    scene_denoising_file = os.path.join(self.path, scene_dir, 'denoising_time.txt')
                    if os.path.exists(scene_denoising_file):
                        try:
                            with open(scene_denoising_file, 'r') as f:
                                scene_time = float(f.read().strip())
                                total_denoising_time += scene_time
                                found_count += 1
                        except Exception as e:
                            print(f"Warning: Failed to read denoising_time.txt from {scene_denoising_file}: {e}")
                
                if found_count > 0:
                    denoising_time = total_denoising_time
        
        # Write time.txt with both total time and denoising time (if available)
        with open(f"{self.path}/time.txt", 'w') as f:
            f.write(f"generate_time: {total_time:.4f}\n")
            if denoising_time is not None:
                f.write(f"denoise_time: {denoising_time:.4f}\n")
        
        fps = self.config.get('fps', 10)
        frames = []
        for frame in all_interpframes:
            if isinstance(frame, Image.Image):
                pass
            elif isinstance(frame, torch.Tensor): # [3, h, w] (0, 1)
                frame = ToPILImage()(frame)
            frames.append(frame)

        if len(frames) == 0:
            os.makedirs(f"{self.path}/frames", exist_ok=True)
            os.makedirs(f"{self.path}/videos", exist_ok=True)
            marker = os.path.join(self.path, "generation_failed.txt")
            if not os.path.exists(marker):
                with open(marker, "w", encoding="utf-8") as f:
                    f.write("No frames generated. Please check MoGE availability or generation logs.\n")
            print("Warning: No frames generated. Skipping frame/video save.")
            return

        save_frames(frames, save_dir=f"{self.path}/frames")
        merge_video(frames, save_dir=f"{self.path}/videos", fps=fps)

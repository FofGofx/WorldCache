# WorldScore/world_generators/voyager.py

import os
import sys
import json
import subprocess
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from PIL import Image
import imageio

# Set Voyager root (env override first)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOYAGER_ROOT = REPO_ROOT / "models" / "HunyuanWorld-Voyager"
VOYAGER_ROOT = Path(os.getenv("VOYAGER_ROOT", str(DEFAULT_VOYAGER_ROOT)))
print(f"Voyager root path set to: {VOYAGER_ROOT}")

sys.path.insert(0, str(VOYAGER_ROOT))
sys.path.insert(0, str(VOYAGER_ROOT / "voyager"))
sys.path.insert(0, str(VOYAGER_ROOT / "voyager" / "worldscore_utils"))

MOGE_ROOT = str(VOYAGER_ROOT / "MoGe")
if MOGE_ROOT not in sys.path:
    sys.path.insert(0, MOGE_ROOT)


class Voyager:
    """Voyager model wrapper for WorldScore."""

    def __init__(
            self,
            model_name: str,
            generation_type: str,
            voyager_root: str,
            model_base: str,
            resolution: List[int],
            frames: int = 49,
            fps: int = 8,
            infer_steps: int = 50,
            cfg_scale: float = 6.0,
            embedded_cfg_scale: float = 6.0,
            flow_shift: float = 7.0,
            seed: int = 0,
            use_cpu_offload: bool = True,
            **kwargs
    ):
        self.model_name = model_name
        self.generation_type = generation_type
        self.voyager_root = Path(voyager_root)
        self.model_base = Path(model_base)
        self.resolution = resolution  # [H, W]
        self.frames = frames
        self.fps = fps
        self.infer_steps = infer_steps
        self.cfg_scale = cfg_scale
        self.embedded_cfg_scale = embedded_cfg_scale
        self.flow_shift = flow_shift
        self.seed = seed
        self.use_cpu_offload = use_cpu_offload
        self.moge_enabled = self._env_flag("WORLDCACHE_ENABLE_MOGE", default=True)
        self.moge_required = self._env_flag("WORLDCACHE_MOGE_REQUIRED", default=False)
        self.moge_model = None

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.create_input_script = self.voyager_root / "data_engine" / "create_input.py"
        self.sample_script = self.voyager_root / "sample_image2video.py"

        self._load_moge()

        self.current_output_dir = None

        print("? Voyager initialized")
        print(f"  Root: {self.voyager_root}")
        print(f"  Resolution: {self.resolution}")
        print(f"  Frames: {self.frames}, FPS: {self.fps}")

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _mark_generation_failed(self, output_dir: Path, reason: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frames").mkdir(exist_ok=True)
        (output_dir / "videos").mkdir(exist_ok=True)
        marker = output_dir / "generation_failed.txt"
        if not marker.exists():
            marker.write_text(reason, encoding="utf-8")
        print(f"? Generation failed: {reason}")

    def _load_moge(self):
        """Load MoGE depth estimation model."""
        if not self.moge_enabled:
            print("? MoGE disabled by WORLDCACHE_ENABLE_MOGE. Running without MoGE.")
            self.moge_model = None
            return
        try:
            if MOGE_ROOT not in sys.path:
                sys.path.insert(0, MOGE_ROOT)
            from moge.model.v1 import MoGeModel
            self.moge_model = MoGeModel.from_pretrained(
                "Ruicheng/moge-vitl",
                local_files_only=True
            ).to("cuda")
            self.moge_model.eval()
            print("? MoGE model loaded")
        except Exception as e:
            print(f"? Failed to load MoGE: {e}")
            self.moge_model = None
            if self.moge_required:
                raise RuntimeError(
                    "MoGE is required but failed to load. "
                    "Please install MoGE or set WORLDCACHE_MOGE_REQUIRED=0."
                ) from e
            print(
                "? MoGE unavailable -> fallback mode enabled. "
                "To enable MoGE, install it or add its path to PYTHONPATH."
            )

    def _load_camera_data(self, output_dir: Path):
        """Load camera data from output_dir."""
        camera_data_file = output_dir / "voyager_camera_data.json"

        if not camera_data_file.exists():
            print(f"? Camera data file not found: {camera_data_file}")
            return None

        with open(camera_data_file, "r") as f:
            data = json.load(f)

        return {
            "cameras_voyager": [np.array(cam) for cam in data["cameras_voyager"]],
            "cameras_interp_voyager": [np.array(cam) for cam in data["cameras_interp_voyager"]],
            "intrinsics": [np.array(K) for K in data["intrinsics"]],
            "camera_path": data["camera_path"],
            "anchor_frame_idx": data["anchor_frame_idx"],
            "num_scenes": data["num_scenes"],
            "resolution": data["resolution"],
            "focal_length": data["focal_length"],
        }

    def generate_video(
            self,
            prompt: str,
            image_path: Optional[str] = None,
            **kwargs
    ) -> List[Image.Image]:
        """
        Generate video for a single prompt instance.
        """
        if image_path is None:
            raise ValueError("Voyager requires an input image (i2v model)")

        output_dir = Path(image_path).parent
        self.current_output_dir = output_dir

        if self.moge_model is None:
            self._mark_generation_failed(
                output_dir,
                "MoGE model not loaded. Worldcache fallback skipped generation."
            )
            return []

        camera_data = self._load_camera_data(output_dir)

        if camera_data is None:
            print("? Using default camera trajectory")
            return self._generate_with_default_camera(prompt, image_path, output_dir)

        image_name = Path(image_path).name

        if image_name == "input_image.png":
            scene_idx = 0
        elif "_" in image_name:
            try:
                scene_idx = int(image_name.split("_")[-1].split(".")[0])
            except ValueError:
                print(f"? Failed to parse scene index from {image_name}, using 0")
                scene_idx = 0
        else:
            print(f"? Unknown image name format: {image_name}, using scene_idx=0")
            scene_idx = 0

        print(f"\n{'='*60}")
        print(f"Voyager.generate_video() - Scene {scene_idx + 1}/{camera_data['num_scenes']}")
        print(f"{'='*60}")
        print(f"  Image: {image_name}")
        print(f"  Prompt: {prompt}")
        print(f"  Scene index: {scene_idx}")

        return self._generate_single_scene_with_index(
            prompt, image_path, output_dir, camera_data, scene_idx
        )

    def _generate_with_default_camera(
            self,
            prompt: str,
            image_path: str,
            output_dir: Path
    ) -> List[Image.Image]:
        """Fallback generation using default camera trajectory."""
        print(f"\n{'='*60}")
        print("Generating with default camera (forward)")
        print(f"{'='*60}")

        input_condition_dir = output_dir / "input_condition"
        input_condition_dir.mkdir(exist_ok=True)

        cmd = [
            "python", str(self.create_input_script),
            "--image_path", image_path,
            "--render_output_dir", str(input_condition_dir),
            "--type", "forward"
        ]
        subprocess.run(cmd, check=True)

        video_output_path = output_dir / "output.mp4"
        self._run_voyager_inference(input_condition_dir, prompt, video_output_path)

        frames = self._extract_rgb_frames(video_output_path)
        return frames

    def _generate_single_scene_with_index(
            self,
            prompt: str,
            image_path: str,
            output_dir: Path,
            camera_data: dict,
            scene_idx: int
    ) -> List[Image.Image]:
        """Generate a scene by index."""
        anchor_frame_idx = camera_data["anchor_frame_idx"]
        num_scenes = camera_data["num_scenes"]

        if scene_idx < 0 or scene_idx >= num_scenes:
            print(f"? Invalid scene_idx {scene_idx}, clamping to [0, {num_scenes-1}]")
            scene_idx = max(0, min(scene_idx, num_scenes - 1))

        start_idx = anchor_frame_idx[scene_idx]
        end_idx = anchor_frame_idx[scene_idx + 1]

        print(f"  Camera trajectory: frames [{start_idx}:{end_idx}] (total {end_idx - start_idx + 1} frames)")

        scene_cameras = camera_data["cameras_interp_voyager"][start_idx:end_idx + 1]
        scene_intrinsics = camera_data["intrinsics"][start_idx:end_idx + 1]

        scene_dir = output_dir / f"scene_{scene_idx}"
        scene_dir.mkdir(exist_ok=True)

        input_condition_dir = scene_dir / "input_condition"
        input_condition_dir.mkdir(exist_ok=True)

        self._create_input_from_cameras(
            image_path,
            scene_cameras,
            scene_intrinsics,
            input_condition_dir
        )

        video_output_path = scene_dir / "output.mp4"
        self._run_voyager_inference(input_condition_dir, prompt, video_output_path)

        frames = self._extract_rgb_frames(video_output_path)

        next_scene_idx = scene_idx + 1
        if next_scene_idx < num_scenes:
            last_frame_path = output_dir / f"input_image_{next_scene_idx}.png"
            frames[-1].save(last_frame_path)
            print(f"? Saved input for next scene: {last_frame_path}")
        else:
            print("? Last scene completed, no more inputs to save")

        return frames

    def _create_input_from_cameras(
            self,
            image_path: str,
            cameras_w2c: List[np.ndarray],
            intrinsics: List[np.ndarray],
            output_dir: Path
    ):
        """Create input condition using custom cameras."""
        from input_creator import VoyagerInputCreator

        if self.moge_model is None:
            self._mark_generation_failed(
                output_dir.parent if output_dir.name == "input_condition" else output_dir,
                "MoGE model not loaded. Input condition generation skipped."
            )
            return

        creator = VoyagerInputCreator(self.moge_model, device="cuda")

        expected_frames = len(cameras_w2c)
        print(f"  Creating input conditions with {expected_frames} camera frames")

        creator.create_input_from_cameras(
            image_path=image_path,
            cameras_w2c=cameras_w2c,
            intrinsics=intrinsics,
            output_dir=str(output_dir),
            width=self.resolution[1],
            height=self.resolution[0]
        )

    def _run_voyager_inference(
            self,
            input_dir: Path,
            prompt: str,
            output_path: Path
    ):
        """Run Voyager inference."""
        dit_weight = self.model_base / "Voyager" / "transformers" / "mp_rank_00_model_states.pt"
        cmd = [
            "python", str(self.sample_script),
            "--model", "HYVideo-T/2",
            "--input-path", str(input_dir),
            "--prompt", prompt,
            "--model-base", str(self.model_base),
            "--i2v-dit-weight", str(dit_weight),
            "--i2v-stability",
            "--infer-steps", str(self.infer_steps),
            "--flow-reverse",
            "--flow-shift", str(self.flow_shift),
            "--seed", str(self.seed),
            "--embedded-cfg-scale", str(self.embedded_cfg_scale),
            "--save-path", str(output_path.parent),
            "--reproduce",
        ]

        if self.use_cpu_offload:
            cmd.append("--use-cpu-offload")

        print("\n  Running Voyager inference...")
        print(f"    Prompt: {prompt}")
        print(f"    Output: {output_path}")

        subprocess.run(cmd, check=True)

    def _extract_rgb_frames(
            self,
            video_path: Path
    ) -> List[Image.Image]:
        """Extract RGB frames from Voyager output video."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        reader = imageio.get_reader(video_path)
        frames = []

        for frame in reader:
            img = Image.fromarray(frame)
            frames.append(img)

        reader.close()

        print(f"  Extracted {len(frames)} frames from video")

        if len(frames) != self.frames:
            print(f"? Expected {self.frames} frames, got {len(frames)}")

        return frames

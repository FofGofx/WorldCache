#WorldScore/models/HunyuanWorld-Voyager/sample_image2video.py
import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from voyager.utils.file_utils import save_videos_grid
from voyager.config import parse_args
from voyager.inference import HunyuanVideoSampler


def _build_model_base_candidates(model_base_arg: str) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(candidate: str | Path | None) -> None:
        if not candidate:
            return
        path = Path(candidate)
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    add(model_base_arg)
    add(os.getenv("MODEL_BASE"))

    model_path = os.getenv("MODEL_PATH")
    if model_path:
        add(Path(model_path) / "HunyuanWorld-Voyager" / "ckpts")

    worldscore_path = os.getenv("WORLDSCORE_PATH")
    if worldscore_path:
        add(Path(worldscore_path) / "models" / "HunyuanWorld-Voyager" / "ckpts")

    add(Path(__file__).resolve().parent / "ckpts")
    return candidates


def _resolve_models_root(args) -> Path:
    candidates = _build_model_base_candidates(args.model_base)
    for candidate in candidates:
        if candidate.exists():
            if str(candidate) != str(args.model_base):
                print(f"[warning] model_base not found, fallback to: {candidate}")
            return candidate

    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise ValueError(
        "models_root not exists. Tried:\n"
        f"{attempted}\n"
        "Set --model-base or env MODEL_BASE / MODEL_PATH / WORLDSCORE_PATH."
    )


def _maybe_fix_i2v_weight(args, models_root_path: Path) -> None:
    if args.i2v_dit_weight:
        weight_path = Path(args.i2v_dit_weight)
        if weight_path.exists():
            return

    fallback_weight = (
        models_root_path / "Voyager" / "transformers" / "mp_rank_00_model_states.pt"
    )
    if fallback_weight.exists():
        args.i2v_dit_weight = str(fallback_weight)
        print(f"[warning] i2v_dit_weight not found, fallback to: {args.i2v_dit_weight}")


def main():
    args = parse_args()
    print("===== Args =====")
    print(args)
    models_root_path = _resolve_models_root(args)
    args.model_base = str(models_root_path)
    _maybe_fix_i2v_weight(args, models_root_path)
    print(f"here, models_root_path: {models_root_path}")

    # Create save folder to save the samples
    save_path = Path(args.save_path) if args.save_path_suffix == "" else Path(
        f"{args.save_path}_{args.save_path_suffix}"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)

    # Get the updated args
    args = hunyuan_video_sampler.args

    # check print
    print(f"[driver] args.video_length={args.video_length}, args.video_size={args.video_size}, fps=8")
    # Start sampling
    # TODO: batch inference check
    input_dir = Path(args.input_path)
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
        i2v_mode=args.i2v_mode,
        i2v_resolution=args.i2v_resolution,
        i2v_image_path=args.i2v_image_path,
        i2v_condition_type=args.i2v_condition_type,
        i2v_stability=args.i2v_stability,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        ref_images=[(str(input_dir / "ref_image.png"),
                     str(input_dir / "ref_depth.exr"))],
        partial_cond=[(str(input_dir / "video_input" / f"render_{j:04d}.png"),
                       str(input_dir / "video_input" / f"depth_{j:04d}.exr")) for j in range(49)],
        partial_mask=[(str(input_dir / "video_input" / f"mask_{j:04d}.png"),
                       str(input_dir / "video_input" / f"mask_{j:04d}.png")) for j in range(49)],
        output_dir=str(save_path)
    )
    samples = outputs['samples']

    # Save generated videos to disk
    # Only save on the main process in distributed settings
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
        # 简化为固定文件名
            cur_save_path = save_path / "output.mp4"
            save_videos_grid(sample, str(cur_save_path), fps=8)
            logger.info(f'Sample save to: {cur_save_path}')



if __name__ == "__main__":
    main()

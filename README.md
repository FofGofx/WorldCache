# WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching

<p align="center">
  <img src="assets/teaser.png" width="90%"/>
</p>

We propose **WorldCache**, a caching framework tailored to diffusion world models. We introduce *Curvature-guided Heterogeneous Token Prediction*, which uses a physics-grounded curvature score to estimate token predictability and applies a Hermite-guided damped predictor for chaotic tokens with abrupt direction changes. We also design *Chaotic-prioritized Adaptive Skipping*, which accumulates a curvature-normalized, dimensionless drift signal and recomputes only when bottleneck tokens begin to drift. Experiments on diffusion world models show that WorldCache delivers up to **3.7×** end-to-end speedups while maintaining **98%** rollout quality, demonstrating the vast advantages and practicality of WorldCache in resource-constrained scenarios.

---

## 🔥 News

- **2026/03** We released the code.

## 🔨 Installation

1. **WorldScore**  
   Follow [WorldScore](https://github.com/haoyi-duan/WorldScore.git) to download the video world model evaluation dataset and code, and configure the evaluation environment. We recommend first running and verifying the WorldScore codebase, then migrating our adaptations for the Voyager and Aether models on WorldScore. The WorldScore-related code and adaptations in this repo live under `worldscore/`.

2. **Voyager**  
   Follow [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager.git) to set up the Voyager model environment, and place the code at **`models/HunyuanWorld-Voyager`**.

3. **Aether**  
   Follow [Aether](https://github.com/InternRobotics/Aether.git) to set up the Aether model environment, and place the code at **`models/Aether`**.

**Environment variables.** Scripts and configs rely on the following; set them before running:

| Variable | Description |
|----------|-------------|
| `WORLDSCORE_PATH` | Root path of this repo / WorldScore (see `config/base_config.yaml`, `worldscore/benchmark/helpers/__init__.py`) |
| `DATA_PATH` | Dataset root (e.g. directory containing `WorldScore-Dataset`; see `dataset_root` in base config) |
| `MODEL_PATH` | Model root; must contain `Aether` and `HunyuanWorld-Voyager` (see `config/model_configs/aether.yaml`, `world_generators/configs/voyager.yaml`) |

## 🚀 Inference

Each script runs video generation with the corresponding conda environment (voyager or aether), then switches to the worldscore environment to run evaluation.

**Voyager:**

```bash
bash scripts/run_voyager_with_worldcache.sh <GPU_ID> <percentile_stable> <percentile_chaotic> <n_max> <error_threshold>
```

Example:

```bash
bash scripts/run_voyager_with_worldcache.sh 0 0.30 0.70 6 1.0
```

**Aether:**

```bash
bash scripts/run_aether_with_worldcache.sh <GPU_ID> <percentile_stable> <percentile_chaotic> <n_max> <error_threshold>
```

Example:

```bash
bash scripts/run_aether_with_worldcache.sh 0 0.20 0.80 2 0.2
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `GPU_ID` | GPU index to use |
| `percentile_stable` | WorldCache stable percentile (e.g. 0.30) |
| `percentile_chaotic` | WorldCache chaotic percentile (e.g. 0.70) |
| `n_max` | WorldCache `n_max` (e.g. 6) |
| `error_threshold` | WorldCache error threshold (e.g. 1.0) |

## 👍 Acknowledgements

Our work is built upon [WorldScore](https://github.com/haoyi-duan/WorldScore.git), [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager.git), [Aether](https://github.com/InternRobotics/Aether.git), [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer.git), [EasyCache](https://github.com/H-EmbodVis/EasyCache.git), [HiCache](https://github.com/fenglang918/HiCache.git), and others. We thank the authors for open-sourcing their code and for their contributions to the community.

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{worldcache2026,
  title     = {WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching},
  author    = {...},
  year      = {2026},
  ...
}
```

(BibTeX / paper link coming soon.)

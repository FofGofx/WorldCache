# WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching

<p>
<a href='https://arxiv.org/abs/2603.06331'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="90%"/>
</p>

We propose **WorldCache**, a caching framework tailored to diffusion world models. We introduce *Curvature-guided Heterogeneous Token Prediction*, which uses a physics-grounded curvature score to estimate token predictability and applies a Hermite-guided damped predictor for chaotic tokens with abrupt direction changes. We also design *Chaotic-prioritized Adaptive Skipping*, which accumulates a curvature-normalized, dimensionless drift signal and recomputes only when bottleneck tokens begin to drift. Experiments on diffusion world models show that WorldCache delivers up to **3.7×** end-to-end speedups while maintaining **98%** rollout quality, demonstrating the vast advantages and practicality of WorldCache in resource-constrained scenarios.

---

## 🔥 News

- **2026/03** Released the code and paper 🚀📄

## 🔨 Installation

1. **WorldScore**  
   Follow [WorldScore](https://github.com/haoyi-duan/WorldScore.git) to download the video world model evaluation dataset and code, and configure the evaluation environment. We recommend first running and verifying the WorldScore codebase, then migrating our adaptations for the Voyager and Aether models on WorldScore.

2. **Voyager**  
   Follow [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager.git) to set up the Voyager model environment, and place the code at **`models/HunyuanWorld-Voyager`**.

3. **Aether**  
   Follow [Aether](https://github.com/InternRobotics/Aether.git) to set up the Aether model environment, and place the code at **`models/Aether`**.

**Environment variables.** Before running any scripts, please make sure the following environment variables are properly set:

- `WORLDSCORE_PATH` — Root path of this repo.
- `DATA_PATH` — Root directory of the evaluation dataset.
- `MODEL_PATH` — Root directory of the models; it should contain both `Aether` and `HunyuanWorld-Voyager`.

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

## 👍 Acknowledgements

Our work is built upon [WorldScore](https://github.com/haoyi-duan/WorldScore.git), [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager.git), [Aether](https://github.com/InternRobotics/Aether.git), [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer.git), [EasyCache](https://github.com/H-EmbodVis/EasyCache.git), [HiCache](https://github.com/fenglang918/HiCache.git), and others. We thank the authors for open-sourcing their code and for their contributions to the community.

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{feng2026worldcacheacceleratingworldmodels,
      title={WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching}, 
      author={Weilun Feng and Guoxin Fan and Haotong Qin and Chuanguang Yang and Mingqiang Wu and Yuqi Li and Xiangqi Li and Zhulin An and Libo Huang and Dingrui Wang and Longlong Liao and Michele Magno and Yongjun Xu},
      year={2026},
      eprint={2603.06331},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.06331}, 
}
```

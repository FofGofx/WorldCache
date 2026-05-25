# WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching

<p align="center">
<a href='https://arxiv.org/abs/2603.06331'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
</p>

<div align="center">
Weilun Feng<sup>*1,2</sup>, Guoxin Fan<sup>*1,2</sup>, Haotong Qin<sup>*3</sup>, Chuanguang Yang<sup>†1</sup>, Mingqiang Wu<sup>1,2</sup>, Yuqi Li<sup>4</sup>, Xiangqi Li<sup>1,2</sup>, Zhulin An<sup>†1</sup>, Libo Huang<sup>1</sup>, Dingrui Wang<sup>5</sup>, Longlong Liao<sup>6</sup>, Michele Magno<sup>3</sup>, Yongjun Xu<sup>1</sup>
</div>

<sup>*</sup>Equal Contribution  <sup>†</sup>Corresponding Author

<div align="center">
1.Institute of Computing Technology, Chinese Academy of Sciences, 2.University of Chinese Academy of Sciences, 3.ETH Zürich, 4.City College of New York, City University of New York, USA, 5.Technical University of Munich, 6.Fuzhou University
</div>

<p align="center">
  <img src="assets/teaser.png" width="90%"/>
</p>

We propose **WorldCache**, a caching framework tailored to diffusion world models. We introduce *Curvature-guided Heterogeneous Token Prediction*, which uses a physics-grounded curvature score to estimate token predictability and applies a Hermite-guided damped predictor for chaotic tokens with abrupt direction changes. We also design *Chaotic-prioritized Adaptive Skipping*, which accumulates a curvature-normalized, dimensionless drift signal and recomputes only when bottleneck tokens begin to drift. Experiments on diffusion world models show that WorldCache delivers up to **3.7×** end-to-end speedups while maintaining **98%** rollout quality, demonstrating the vast advantages and practicality of WorldCache in resource-constrained scenarios.

---

## 🔥 News

- **2026/05** WorldCache is now adapted to **LingBot-World-Base (Cam)** ✨✨✨
- **2026/05** Paper accepted by ICML 2026 🎉🎉🎉
- **2026/03** Released the code and paper 🚀🚀🚀

## 🔨 Installation

1. **WorldScore**  
   Follow [WorldScore](https://github.com/haoyi-duan/WorldScore.git) to download the video world model evaluation dataset and code, and configure the evaluation environment. We recommend first running and verifying the WorldScore codebase, then migrating our adaptations for the Voyager and Aether models on WorldScore.

2. **Voyager**  
   Follow [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager.git) to set up the Voyager model environment, and place the code at **`models/HunyuanWorld-Voyager`**.

3. **Aether**  
   Follow [Aether](https://github.com/InternRobotics/Aether.git) to set up the Aether model environment, and place the code at **`models/Aether`**.

4. **LingBot-World-Base (Cam)**  
   Follow [lingbot-world](https://github.com/robbyant/lingbot-world) to set up the LingBot-World environment, and place the code at **`models/lingbot-world`**.

**Environment variables.** Before running any scripts, please make sure the following environment variables are properly set:

- `WORLDSCORE_PATH` — Root path of this repo.
- `DATA_PATH` — Root directory of the evaluation dataset.
- `MODEL_PATH` — Root directory of the models; it should contain `Aether`, `HunyuanWorld-Voyager`, and `lingbot-world`.

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
bash scripts/run_aether_with_worldcache.sh 0 0.30 0.60 2 0.2
```

**LingBot-World-Base (Cam):**

```bash
WORLDCACHE_MODE=worldcache bash scripts/run_lingbot_with_worldcache.sh <GPU_ID> <percentile_stable> <percentile_chaotic> <n_max> <error_threshold>
```

Example:

```bash
WORLDCACHE_MODE=worldcache bash scripts/run_lingbot_with_worldcache.sh 6 0.30 0.60 6 0.6
```

## 🎬 Demo

### LingBot-World-Base (Cam)

GitHub does not render embedded repository videos in `README.md`, so the table below uses GIF previews. Click any preview to open the full MP4.

<table>
  <tr>
    <th>Case</th>
    <th>Original</th>
    <th>WorldCache</th>
  </tr>
  <tr>
    <td>03<br/>SpeedUp: <b>1x vs 2.25x</b></td>
    <td>
      <a href="./models/lingbot-world/outputs/original_03/i2v-A14B_480x832_stepsdefault_framesdefault_original.mp4">
        <img src="assets/demo_original_03.gif" width="240" alt="Case 03 original preview"/>
      </a>
      <br/>
      <sub><a href="./models/lingbot-world/outputs/original_03/i2v-A14B_480x832_stepsdefault_framesdefault_original.mp4">Open MP4</a></sub>
    </td>
    <td>
      <a href="./models/lingbot-world/outputs/worldcache_p30_c60_n6_e60_03/i2v-A14B_480x832_stepsdefault_framesdefault_worldcache_p30_c60_n6_e60.mp4">
        <img src="assets/demo_worldcache_03.gif" width="240" alt="Case 03 WorldCache preview"/>
      </a>
      <br/>
      <sub><a href="./models/lingbot-world/outputs/worldcache_p30_c60_n6_e60_03/i2v-A14B_480x832_stepsdefault_framesdefault_worldcache_p30_c60_n6_e60.mp4">Open MP4</a></sub>
    </td>
  </tr>
  <tr>
    <td>04<br/>SpeedUp: <b>1x vs 2.22x</b></td>
    <td>
      <a href="./models/lingbot-world/outputs/original_04/i2v-A14B_480x832_stepsdefault_framesdefault_original.mp4">
        <img src="assets/demo_original_04.gif" width="240" alt="Case 04 original preview"/>
      </a>
      <br/>
      <sub><a href="./models/lingbot-world/outputs/original_04/i2v-A14B_480x832_stepsdefault_framesdefault_original.mp4">Open MP4</a></sub>
    </td>
    <td>
      <a href="./models/lingbot-world/outputs/worldcache_p30_c60_n6_e60_04/i2v-A14B_480x832_stepsdefault_framesdefault_worldcache_p30_c60_n6_e60.mp4">
        <img src="assets/demo_worldcache_04.gif" width="240" alt="Case 04 WorldCache preview"/>
      </a>
      <br/>
      <sub><a href="./models/lingbot-world/outputs/worldcache_p30_c60_n6_e60_04/i2v-A14B_480x832_stepsdefault_framesdefault_worldcache_p30_c60_n6_e60.mp4">Open MP4</a></sub>
    </td>
  </tr>
</table>

## 👍 Acknowledgements

Our work is built upon [WorldScore](https://github.com/haoyi-duan/WorldScore.git), [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager.git), [Aether](https://github.com/InternRobotics/Aether.git), [lingbot-world](https://github.com/robbyant/lingbot-world), [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer.git), [EasyCache](https://github.com/H-EmbodVis/EasyCache.git), [HiCache](https://github.com/fenglang918/HiCache.git), and others. We thank the authors for open-sourcing their code and for their contributions to the community.

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

#!/bin/bash
# 示例脚本：使用不同的 Worldcache 参数并行运行 Aether 模型生成
# 用法: ./run_aether_with_worldcache.sh <GPU_ID> <percentile_stable> <percentile_chaotic> <n_max> <error_threshold>
#
# 示例:
#   ./run_aether_with_worldcache.sh 0 0.20 0.80 2 0.2
#   ./run_aether_with_worldcache.sh 1 0.15 0.85 3 0.15

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate aether

# 获取参数
GPU_ID=$1
PERCENTILE_STABLE=$2
PERCENTILE_CHAOTIC=$3
N_MAX=$4
ERROR_THRESHOLD=$5

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 设置 Worldcache 环境变量
export WORLDCACHE_PERCENTILE_STABLE=$PERCENTILE_STABLE
export WORLDCACHE_PERCENTILE_CHAOTIC=$PERCENTILE_CHAOTIC
export WORLDCACHE_N_MAX=$N_MAX
export WORLDCACHE_ERROR_THRESHOLD=$ERROR_THRESHOLD

# 生成输出目录后缀
# 格式: _worldscore_p{percentile_stable}_c{percentile_chaotic}_n{n_max}_e{error_threshold}
P_STABLE_INT=$(echo "$PERCENTILE_STABLE * 100" | bc | cut -d. -f1)
P_CHAOTIC_INT=$(echo "$PERCENTILE_CHAOTIC * 100" | bc | cut -d. -f1)
E_THRESHOLD_INT=$(printf "%.0f" $(echo "$ERROR_THRESHOLD * 100" | bc))
E_THRESHOLD_FORMATTED=$(printf "%02d" $E_THRESHOLD_INT)

export WORLDCACHE_OUTPUT_SUFFIX="_worldscore_p${P_STABLE_INT}_c${P_CHAOTIC_INT}_n${N_MAX}_e${E_THRESHOLD_FORMATTED}"

echo "GPU ID: $GPU_ID"
echo "Worldcache 参数:"
echo "  percentile_stable: $PERCENTILE_STABLE"
echo "  percentile_chaotic: $PERCENTILE_CHAOTIC"
echo "  n_max: $N_MAX"
echo "  error_threshold: $ERROR_THRESHOLD"
echo "输出目录后缀: $WORLDCACHE_OUTPUT_SUFFIX"
echo "=========================================="


python world_generators/generate_videos.py \
    --model-name aether \
    --worldcache-percentile-stable $PERCENTILE_STABLE \
    --worldcache-percentile-chaotic $PERCENTILE_CHAOTIC \
    --worldcache-n-max $N_MAX \
    --worldcache-error-threshold $ERROR_THRESHOLD

conda activate worldscore


python worldscore/run_evaluate.py --model_name aether


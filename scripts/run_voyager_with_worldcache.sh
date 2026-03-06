#!/bin/bash
# Example script: run Voyager generation with Worldcache parameters
# Usage: ./run_voyager_with_worldcache.sh <GPU_ID> <percentile_stable> <percentile_chaotic> <n_max> <error_threshold>
#
# Examples:
#   ./run_voyager_with_worldcache.sh 0 0.30 0.70 6 1.0
#   ./run_voyager_with_worldcache.sh 1 0.20 0.80 4 0.5

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate voyager


GPU_ID=$1
PERCENTILE_STABLE=$2
PERCENTILE_CHAOTIC=$3
N_MAX=$4
ERROR_THRESHOLD=$5

export CUDA_VISIBLE_DEVICES=$GPU_ID

export WORLDCACHE_PERCENTILE_STABLE=$PERCENTILE_STABLE
export WORLDCACHE_PERCENTILE_CHAOTIC=$PERCENTILE_CHAOTIC
export WORLDCACHE_N_MAX=$N_MAX
export WORLDCACHE_ERROR_THRESHOLD=$ERROR_THRESHOLD

export WORLDCACHE_MODE=worldcache

P_STABLE_INT=$(awk "BEGIN {printf \"%.0f\", $PERCENTILE_STABLE * 100}")
P_CHAOTIC_INT=$(awk "BEGIN {printf \"%.0f\", $PERCENTILE_CHAOTIC * 100}")
E_THRESHOLD_INT=$(awk "BEGIN {printf \"%.0f\", $ERROR_THRESHOLD * 100}")
E_THRESHOLD_FORMATTED=$(printf "%03d" $E_THRESHOLD_INT)

OUTPUT_DIR_SUFFIX="worldscore_p${P_STABLE_INT}_c${P_CHAOTIC_INT}_n${N_MAX}_e${E_THRESHOLD_FORMATTED}"

export OUTPUT_DIR_SUFFIX="$OUTPUT_DIR_SUFFIX"
export WORLDCACHE_OUTPUT_SUFFIX="_${OUTPUT_DIR_SUFFIX}"

echo "=========================================="
echo "Running Voyager generation"
echo "=========================================="
echo "GPU ID: $GPU_ID"
echo "Worldcache parameters:"
echo "  percentile_stable: $PERCENTILE_STABLE"
echo "  percentile_chaotic: $PERCENTILE_CHAOTIC"
echo "  n_max: $N_MAX"
echo "  error_threshold: $ERROR_THRESHOLD"
echo "Output dir suffix: $OUTPUT_DIR_SUFFIX"
echo "=========================================="

python world_generators/generate_videos.py \
    --model-name voyager \
    --worldcache_percentile_stable "$PERCENTILE_STABLE" \
    --worldcache_percentile_chaotic "$PERCENTILE_CHAOTIC" \
    --worldcache_n_max "$N_MAX" \
    --worldcache_error_threshold "$ERROR_THRESHOLD"

echo "=========================================="
echo "Generation complete"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate worldscore

echo "=========================================="
echo "Starting evaluation"
echo "=========================================="
python worldscore/run_evaluate.py --model_name voyager

echo "=========================================="
echo "Evaluation complete"
echo "=========================================="

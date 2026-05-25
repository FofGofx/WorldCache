#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 <GPU_ID> <percentile_stable> <percentile_chaotic> <n_max> <error_threshold>" >&2
  exit 1
fi

GPU_ID="$1"
PERCENTILE_STABLE="$2"
PERCENTILE_CHAOTIC="$3"
N_MAX="$4"
ERROR_THRESHOLD="$5"

MODE="${WORLDCACHE_MODE:-worldcache}"
if [[ "${MODE}" != "original" && "${MODE}" != "worldcache" ]]; then
  echo "Unsupported WORLDCACHE_MODE=${MODE}. Use 'original' or 'worldcache'." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/models/lingbot-world"

TASK="${LINGBOT_TASK:-i2v-A14B}"
SIZE="${LINGBOT_SIZE:-480*832}"
CKPT_DIR="${LINGBOT_CKPT_DIR:-${MODEL_DIR}/lingbot-world-base-cam}"
IMAGE="${LINGBOT_IMAGE:-${MODEL_DIR}/examples/04/image.jpg}"
ACTION_PATH="${LINGBOT_ACTION_PATH:-${MODEL_DIR}/examples/04}"
OUTPUT_ROOT="${LINGBOT_OUTPUT_ROOT:-${MODEL_DIR}/outputs}"
FRAME_NUM="${LINGBOT_FRAME_NUM:-}"
PROMPT="${LINGBOT_PROMPT:-}"
SAMPLE_STEPS="${LINGBOT_SAMPLE_STEPS:-}"

SIZE_TAG="${SIZE//\*/x}"
FRAME_TAG="${FRAME_NUM:-default}"
STEP_TAG="${SAMPLE_STEPS:-default}"
P_STABLE_INT="$(awk "BEGIN {printf \"%.0f\", ${PERCENTILE_STABLE} * 100}")"
P_CHAOTIC_INT="$(awk "BEGIN {printf \"%.0f\", ${PERCENTILE_CHAOTIC} * 100}")"
E_THRESHOLD_INT="$(awk "BEGIN {printf \"%.0f\", ${ERROR_THRESHOLD} * 100}")"
E_THRESHOLD_TAG="$(printf "%02d" "${E_THRESHOLD_INT}")"

if [[ "${MODE}" == "worldcache" ]]; then
  MODE_TAG="worldcache_p${P_STABLE_INT}_c${P_CHAOTIC_INT}_n${N_MAX}_e${E_THRESHOLD_TAG}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${MODE_TAG}"
else
  MODE_TAG="original"
  OUTPUT_DIR="${OUTPUT_ROOT}/original"
fi

mkdir -p "${OUTPUT_DIR}"
SAVE_FILE="${OUTPUT_DIR}/${TASK}_${SIZE_TAG}_steps${STEP_TAG}_frames${FRAME_TAG}_${MODE_TAG}.mp4"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WORLDCACHE_MODE="${MODE}"
export WORLDCACHE_PERCENTILE_STABLE="${PERCENTILE_STABLE}"
export WORLDCACHE_PERCENTILE_CHAOTIC="${PERCENTILE_CHAOTIC}"
export WORLDCACHE_N_MAX="${N_MAX}"
export WORLDCACHE_ERROR_THRESHOLD="${ERROR_THRESHOLD}"

cd "${MODEL_DIR}"
eval "$(conda shell.bash hook)"
conda activate lingbot-world

CMD=(
  torchrun
  --nproc_per_node=1
  generate.py
  --task "${TASK}"
  --size "${SIZE}"
  --ckpt_dir "${CKPT_DIR}"
  --image "${IMAGE}"
  --action_path "${ACTION_PATH}"
  --save_file "${SAVE_FILE}"
)

if [[ -n "${FRAME_NUM}" ]]; then
  CMD+=(--frame_num "${FRAME_NUM}")
fi

if [[ -n "${SAMPLE_STEPS}" ]]; then
  CMD+=(--sample_steps "${SAMPLE_STEPS}")
fi

if [[ -n "${PROMPT}" ]]; then
  CMD+=(--prompt "${PROMPT}")
fi

echo "=========================================="
echo "Running LingBot-World"
echo "=========================================="
echo "GPU ID: ${GPU_ID}"
echo "Mode: ${MODE}"
echo "Task: ${TASK}"
echo "Size: ${SIZE}"
echo "Checkpoint: ${CKPT_DIR}"
echo "Image: ${IMAGE}"
echo "Action path: ${ACTION_PATH}"
echo "percentile_stable: ${PERCENTILE_STABLE}"
echo "percentile_chaotic: ${PERCENTILE_CHAOTIC}"
echo "n_max: ${N_MAX}"
echo "error_threshold: ${ERROR_THRESHOLD}"
echo "Output file: ${SAVE_FILE}"
echo "=========================================="

"${CMD[@]}"

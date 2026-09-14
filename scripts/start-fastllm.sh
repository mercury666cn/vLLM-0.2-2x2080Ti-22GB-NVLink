#!/bin/bash
set -eu
# shellcheck source=runtime-common.sh
. "$(dirname "$(readlink -f "$0")")/runtime-common.sh"
require_docker

IMAGE="native/fastllm:official-wheel"
CONTAINER="fast-prod-nvfp4-api"
MODEL_NAME="qwen38-fastllm-nvfp4-mtp3-cn-210k"
MODEL_DIR="${MODEL_ROOT}/Qwen3.8-27B-NVFP4"

acquire_lock
echo "Switching to FastLLM NVFP4 MTP3 FP16-KV 240K thinking-off..."
stop_all_gpu_runtimes
assert_single_runtime
wait_gpu_free

docker run -d \
  --name "${CONTAINER}" \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --restart unless-stopped \
  -v "${MODEL_ROOT}:${MODEL_ROOT}:ro" \
  "${IMAGE}" server "${MODEL_DIR}" \
  --model_name "${MODEL_NAME}" \
  --host 127.0.0.1 \
  --port 8001 \
  --device cuda \
  --tp 0,1 \
  --max_batch 1 \
  --gpu_mem_ratio 0.95 \
  --tokens 240000 \
  --max_context_length 240000 \
  --kv_cache_dtype float16 \
  --prefix_cache false \
  --mtp 3 \
  --enable_thinking false

wait_models "${MODEL_NAME}"
print_ready "fastllm" "${MODEL_NAME}"

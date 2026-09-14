#!/bin/bash
set -eu
# shellcheck source=runtime-common.sh
. "$(dirname "$(readlink -f "$0")")/runtime-common.sh"
require_docker

IMAGE="native/vllm-2080ti:v0.2.1-pre3-sm75"
CONTAINER="nvfp4-v02-210k"
MODEL_NAME="qwen38-v02-nvfp4-mtp3-210k"
MODEL_DIR="${MODEL_ROOT}/Qwen3.8-27B-NVFP4"

acquire_lock
echo "Switching to vLLM 0.2 NVFP4 MTP3 210K..."
stop_all_gpu_runtimes
assert_single_runtime
wait_gpu_free

mkdir -p \
  "${RUNTIME_ROOT}/native-logs/v02-nvfp4-210k" \
  "${RUNTIME_ROOT}/vllm-pre3-final2-cache" \
  "${RUNTIME_ROOT}/vllm-pre3-final2-triton"

docker run -d \
  --name "${CONTAINER}" \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --restart unless-stopped \
  -e NO_PROXY=127.0.0.1,localhost \
  -v "${MODEL_ROOT}:${MODEL_ROOT}:ro" \
  -v "${RUNTIME_ROOT}/vllm-pre3-final2-cache:/root/.cache/vllm" \
  -v "${RUNTIME_ROOT}/vllm-pre3-final2-triton:/workspace/triton-cache" \
  -v "${RUNTIME_ROOT}/native-logs/v02-nvfp4-210k:/workspace/run-logs" \
  --entrypoint bash "${IMAGE}" -lc "set -e
ln -sfn /opt/venv /workspace/.venv
cd /workspace
./launcher.sh --non-interactive \
  --set PROFILE=qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-only.env \
  --set MODEL_DIR=${MODEL_DIR} \
  --set PORT=8001 \
  --set MAX_MODEL_LEN=210000 \
  --set SERVED_NAME=qwen38-v02-nvfp4-mtp3-210k \
  --set MAX_NUM_SEQS=1 \
  --set SERVICE_SCOPE=local
pid=\$(cat /workspace/run-logs/vllm-qwen38-v02-nvfp4-mtp3-210k.pid)
echo KEEP_PID=\$pid
while kill -0 \"\$pid\" 2>/dev/null; do sleep 2; done
exit 1"

wait_models "${MODEL_NAME}"
print_ready "v0.2" "${MODEL_NAME}"

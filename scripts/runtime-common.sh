#!/bin/bash
set -eu

# Public copy: set these on your machine. Do not hardcode LAN IPs or host paths.
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$(dirname "${SCRIPT_DIR}")}"
MODEL_ROOT="${MODEL_ROOT:-${PWD}/models}"
LAN_IP="${LAN_IP:-127.0.0.1}"
LOCK_FILE="${SCRIPT_DIR}/.runtime.lock"
ACTIVE_FILE="${SCRIPT_DIR}/active-runtime"

BACKEND_PORT=8001
PUBLIC_PORT=8000
COMPAT_CONTAINER="fast-prod-compat"

KNOWN_CONTAINERS=(
  fast-prod-compat
  fast-prod-nvfp4-api
  nvfp4-v02-210k
  quality-eval-engine
  fast-cn-nvfp4-api
  nvfp4-v02-8001
  nvfp4-8001
  fp8-v02-8001
)

require_docker() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  if [ "$(id -u)" -ne 0 ]; then
    echo "当前用户没有 docker 权限，改用 sudo 重跑..."
    exec sudo "$0" "$@"
  fi
  echo "ERROR: cannot talk to docker" >&2
  exit 1
}

acquire_lock() {
  mkdir -p "${SCRIPT_DIR}"
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "ERROR: another runtime switch is already running" >&2
    exit 4
  fi
}

stop_all_gpu_runtimes() {
  local name cid
  for name in "${KNOWN_CONTAINERS[@]}"; do
    docker rm -f "${name}" >/dev/null 2>&1 || true
  done
  while read -r cid; do
    [ -n "${cid}" ] || continue
    docker rm -f "${cid}" >/dev/null 2>&1 || true
  done < <(docker ps -aq --filter "name=retired-" 2>/dev/null || true)
}

assert_single_runtime() {
  local running
  running="$(docker ps --format '{{.Names}}' | grep -E 'fast-prod-nvfp4-api|nvfp4-v02-210k' || true)"
  if [ -n "${running}" ]; then
    echo "ERROR: unexpected leftover runtime still up:" >&2
    echo "${running}" >&2
    exit 6
  fi
}

wait_gpu_free() {
  local i used
  for i in $(seq 1 60); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print int(s+0)}')"
    if [ "${used}" -lt 800 ]; then
      echo "GPU_FREE used_mib=${used}"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: GPU memory still occupied" >&2
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  exit 5
}

wait_models_port() {
  local port="$1"
  local model="$2"
  local i
  for i in $(seq 1 180); do
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q "${model}"; then
      echo "READY_PORT=${port} READY_SEC=$((i * 2))"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: API did not become ready for ${model} on :${port}" >&2
  docker ps -a --format '{{.Names}} {{.Status}}'
  return 1
}

start_compat_proxy() {
  docker rm -f "${COMPAT_CONTAINER}" >/dev/null 2>&1 || true
  docker run -d \
    --name "${COMPAT_CONTAINER}" \
    --network host \
    --restart unless-stopped \
    -e COMPAT_BACKEND="http://127.0.0.1:${BACKEND_PORT}" \
    -e COMPAT_PORT="${PUBLIC_PORT}" \
    -v "${SCRIPT_DIR}/compat-proxy.py:/opt/compat-proxy.py:ro" \
    --entrypoint python3 \
    native/fastllm:official-wheel \
    /opt/compat-proxy.py
}

wait_models() {
  local model="$1"
  wait_models_port "${BACKEND_PORT}" "${model}"
  start_compat_proxy
  wait_models_port "${PUBLIC_PORT}" "${model}"
  if ! curl -fsS --max-time 5 "http://127.0.0.1:${PUBLIC_PORT}/models" | grep -q "${model}"; then
    echo "ERROR: compat /models alias is not serving ${model}" >&2
    return 1
  fi
  curl -fsS --max-time 5 "http://127.0.0.1:${PUBLIC_PORT}/models"
}

print_ready() {
  local runtime="$1"
  local model="$2"
  printf '%s\n' "${runtime}" >"${ACTIVE_FILE}"
  echo
  echo "===== ${runtime} READY ====="
  echo "API:   http://${LAN_IP}:8000/v1"
  echo "Also:  http://${LAN_IP}:8000/models"
  echo "Chat:  http://${LAN_IP}:8000/v1/chat/completions"
  echo "Model: ${model}"
  docker ps --format '{{.Names}} {{.Status}}' | grep -E 'fast-prod-compat|fast-prod-nvfp4-api|nvfp4-v02-210k' || true
}

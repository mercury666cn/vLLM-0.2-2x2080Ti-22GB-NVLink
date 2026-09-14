#!/bin/bash
set -eu
# shellcheck source=runtime-common.sh
. "$(dirname "$(readlink -f "$0")")/runtime-common.sh"
require_docker

engine="$(docker ps --format '{{.Names}}' | grep -E 'fast-prod-nvfp4-api|nvfp4-v02-210k' | head -n 1 || true)"
container="$(docker ps --format '{{.Names}}' | grep -E 'fast-prod-compat' | head -n 1 || true)"
if [ -z "${container}" ]; then
  container="${engine}"
fi
active="$(cat "${ACTIVE_FILE}" 2>/dev/null || echo unknown)"

echo "===== runtime ====="
echo "active=${active}"
if [ -n "${engine}" ]; then
  docker ps --filter "name=${engine}" --format 'engine={{.Names}}  status={{.Status}}'
else
  echo "engine=(none running)"
fi
if [ -n "${container}" ] && [ "${container}" != "${engine}" ]; then
  docker ps --filter "name=${container}" --format 'compat={{.Names}}  status={{.Status}}'
fi

echo
echo "===== gpu ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

echo
echo "===== listen ====="
ss -lntp | grep -E ':8000|:8001' || echo "(nothing on :8000/:8001)"

echo
echo "===== current clients on :8000 ====="
ss -tnp state established '( sport = :8000 )' || true

echo
echo "===== last 40 access lines ====="
if [ -n "${container}" ]; then
  echo "tip: live follow -> watch-runtime"
  docker logs --tail 40 "${container}"
else
  echo "(no container)"
fi

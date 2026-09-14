#!/bin/bash
set -eu
# shellcheck source=runtime-common.sh
. "$(dirname "$(readlink -f "$0")")/runtime-common.sh"
require_docker

container="$(docker ps --format '{{.Names}}' | grep -E 'fast-prod-compat' | head -n 1 || true)"
if [ -z "${container}" ]; then
  container="$(docker ps --format '{{.Names}}' | grep -E 'fast-prod-nvfp4-api|nvfp4-v02-210k' | head -n 1 || true)"
fi
if [ -z "${container}" ]; then
  echo "ERROR: no production runtime is running" >&2
  exit 1
fi

echo "Following ${container}  (Ctrl+C to stop)"
echo "Each line is one HTTP hit: client_ip - \"METHOD /path\" status"
echo
docker logs -f --since 5m "${container}"

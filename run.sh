#!/usr/bin/env bash
set -euo pipefail

IMAGE=newton-examples
HERE="$(cd "$(dirname "$0")" && pwd)"
NEWTON_WS="$HERE/newton_ws"
CONTAINER_NAME="newton-lab-Mutasci"

# Cartella del tuo progetto sull'host
mkdir -p "$NEWTON_WS"

# Permette al container di usare il display X11 dell'host
xhost +local:docker >/dev/null

# Senza argomenti apre una shell, altrimenti esegue il comando passato, es.:
#   ./run.sh uv run -m newton.examples basic_pendulum
#
# - newton_ws (host)  ->  /workspace/project_workspace (container): il tuo progetto
# - /workspace/newton (repo + .venv) resta quello dell'immagine, non mappato
docker run --name "$CONTAINER_NAME" --rm -it --gpus all \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$NEWTON_WS":/workspace/project_workspace \
  "$IMAGE" "$@"

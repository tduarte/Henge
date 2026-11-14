#!/usr/bin/env bash
# Start a native ComfyUI on Apple Silicon (MPS) the way Henge expects it.
# Usage: COMFY_DIR=/path/to/ComfyUI scripts/start_comfy_mac.sh [extra ComfyUI args]
#
# --use-pytorch-cross-attention: ComfyUI's default sub-quadratic attention on MPS can corrupt heads at H3's long
#   sequence lengths (dark video, ComfyUI issue #17418); SDPA does not.
# No --lowvram: unified memory is shared with the CPU, ComfyUI picks its SHARED vram state on MPS by itself.
# PYTORCH_ENABLE_MPS_FALLBACK: ops MPS lacks run on the CPU instead of failing the job.
set -euo pipefail
COMFY_DIR=${COMFY_DIR:?set COMFY_DIR to your ComfyUI folder}
PYTHON=${PYTHON:-$COMFY_DIR/.venv/bin/python}
PORT=${COMFY_PORT:-8189}
[ "$(uname -s)" = Darwin ] || { echo "This launcher is for macOS; see docs/RUN_LOCAL.md"; exit 1; }
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
export HENGE_PLATFORM=${HENGE_PLATFORM:-auto}
cd "$COMFY_DIR"
exec "$PYTHON" main.py --listen 127.0.0.1 --port "$PORT" --preview-method taesd \
  --use-pytorch-cross-attention "$@"

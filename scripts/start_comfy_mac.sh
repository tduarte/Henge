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
# On MPS ComfyUI always uses its SHARED memory state, which runs text encoders (the 32B H3 encoder and the 9B planner)
# on the CPU: many times slower. --gpu-only keeps them on the Apple GPU. Default on with 96 GB or more of unified
# memory (bf16 H3 + its encoder + the planner fit); HENGE_GPU_ONLY=0/1 overrides.
mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
extra=()
if [ "${HENGE_GPU_ONLY:-$([ "$mem_gb" -ge 96 ] && echo 1 || echo 0)}" = 1 ]; then extra+=(--gpu-only); fi
cd "$COMFY_DIR"
exec "$PYTHON" main.py --listen 127.0.0.1 --port "$PORT" --preview-method taesd \
  --use-pytorch-cross-attention ${extra[@]+"${extra[@]}"} "$@"

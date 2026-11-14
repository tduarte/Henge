# Henge on Apple Silicon (native, MPS)

Henge's local option assumes an NVIDIA GPU. This fork also runs the UI, the bridge and ComfyUI natively on an
Apple Silicon Mac, with PyTorch's MPS backend and no containers. **This is experimental:** it is slower than an
NVIDIA GPU and each mode still needs checking end to end on a Mac.

> **Why not Docker or Apple's `container`?** Both run Linux in a VM with no access to Metal, so ComfyUI would
> run on the CPU only. ComfyUI has to run natively. The UI and bridge can go anywhere.

## What changes on a Mac

| Published recipe (NVIDIA) | On MPS | Where |
|---|---|---|
| Comfy Kitchen INT8 attention (`ModelAttentionBackend`) | `pytorch attention` (ComfyUI only offers that one on MPS) | `backend/platform_compat.py` |
| TARA kernel / Kitchen temporal backend | `flash partitions (no compile)`, using exact chunked attention with LSE in plain PyTorch | `platform_compat.py`, ComfyUI-BFSNodes `minimax_h3_temporal_fast.py` (fork) |
| Sol sparse attention (`BlockSparseAttention`, Triton) | node bypassed: dense SDPA (same result, slower) | `platform_compat.py` |
| `MinimaxH3LatentUpscaler3D device=cuda` | the MPS device | `platform_compat.py` |
| 3D body render on `cuda` | ComfyUI's torch device | `comfy_nodes/henge/mesh_renderer.py` |
| Upscaler cache flush / device check, CUDA only | CUDA, MPS or CPU | `comfy_nodes/henge/upscale.py` |
| `ss`, `grep -P` in `local_stack.sh` | `lsof` when `ss` is missing | `scripts/local_stack.sh` |
| NVIDIA RTX VSR upscaler | not available; use `h3`, `regenerate` or `none` | (hidden automatically) |

The bridge reads ComfyUI's device from `/system_stats` before each job. `HENGE_PLATFORM=cuda|mps|cpu` overrides
that. A CUDA backend gets every graph unchanged.

## Model quantizations

The graphs still name the published files (`*_int8_convrot`). For each UNET, CLIP or VAE loader, the bridge uses
whichever installed file is **the same model in another quantization** (`platform_compat.quant_sibling`). It
prefers formats in the order given by `HENGE_QUANT_PREFERENCE`, which defaults to
`bf16,fp16,q8_0,int8_convrot,nvfp4_awq,fp32`. LoRAs are never swapped. The setup checklist and `/health` accept
the sibling too.

| Format | On MPS |
|---|---|
| bf16 / fp16 | native; fastest |
| int8_convrot | works (ComfyUI dequantizes each layer through comfy_kitchen's eager backend); slower per step |
| nvfp4_awq | works, emulated; reported to be slower than int8 |
| **fp8 (e4m3fn, `fp8_scaled`)** | **fails**: MPS cannot cast float8. Avoid it |
| GGUF | needs a ComfyUI-GGUF loader with the `minimax` architecture (leejet's fork). When only a GGUF copy is installed, the bridge switches the node to `UnetLoaderGGUF`/`CLIPLoaderGGUF` |
| int8 **video VAE** | crashes on MPS (ComfyUI PR #16890 is open). Keep `minimax_h3_video_vae_fp16` |

`config/model-repos.apple-silicon.json` is the manifest for a 128 GB Mac:

- H3 base: `minimax_h3_ref2va_pruned_bf16` (40.2 GB)
- H3 text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq` (15.7 GB). Use `…_int8_convrot` (27.1 GB) if nvfp4 fails.
- Planner: `qwen3.5_9b_bf16`
- FLUX.2 Klein is left out because it is fp8 only.

Don't load the bf16 text encoder (51.5 GB) together with the bf16 base: that leaves too little memory for video.
On 64 GB, use the int8 base (21 GB) or the GGUF Q8_0 base with the GGUF Q4 encoder.

```sh
python scripts/download_models.py --manifest config/model-repos.apple-silicon.json \
  --models-dir /path/to/ComfyUI/models --profiles character,motion_transfer,motion_transfer_tara,sam2d,audio
```

## Run it

1. Install ComfyUI natively in a venv: `uv venv --python 3.13`, then `uv pip install torch torchvision
   torchaudio`, then `uv pip install -r requirements.txt`.
2. Link the nodes as in [RUN_LOCAL.md](RUN_LOCAL.md). Use the BFSNodes fork with the MPS attention fallback,
   and install `backend/requirements.txt` into the same venv.
3. Start ComfyUI with `COMFY_DIR=/path/to/ComfyUI scripts/start_comfy_mac.sh`. The script runs on port 8189 with
   `--use-pytorch-cross-attention`. ComfyUI's default sub-quadratic attention on MPS can produce dark video at H3
   sequence lengths (ComfyUI #17418). The script also sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, and it doesn't pass
   `--lowvram`.
4. Create `.env.local` (`COMFY_BASE_URL=http://127.0.0.1:8189`, `COMFY_ROOT`, `HENGE_DATA_DIR`), then run
   `npm ci && npm run build` and `PYTHON=/path/to/ComfyUI/.venv/bin/python scripts/local_stack.sh start`.

## Known limits

- Speed: Mac users report minutes per step at 480p. Start with short clips, 480p and the 4-step turbo LoRAs.
- ComfyUI #16673 (open): multi-frame video guides can be ignored on MPS. Check the guide actually follows.
- Molmo pointing (optional) still loads on `cuda` with an FP8 checkpoint, so it is unavailable on a Mac.
- The exported workflow JSONs under `workflows/` still say `device: cuda` for the latent upscaler. Change it to
  `mps` when opening one directly in ComfyUI. Graphs the bridge sends are already adapted.

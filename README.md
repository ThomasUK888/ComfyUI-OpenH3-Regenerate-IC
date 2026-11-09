# ComfyUI-OpenH3-Regenerate-IC

ComfyUI nodes for NVIDIA's [Open-H3-Regenerate-2K-IC](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC):
regenerate a low-resolution MiniMax-H3 draft (1344x768) at 1080p or 2K **while keeping its faces,
expressions and layout**. The draft rides in context as `<Video 1>` of a Ref2VA request; NVIDIA's IC LoRA
generates the same clip at the target size.

Runs on a single 32 GB card (tested on an RTX 5090) with ComfyUI's native MiniMax-H3 support.

## What it is good for

Measured on FastH3 720p drafts, 5090, 124-frame (5.2 s) clips:

| | time | |
|---|---|---|
| 1080p, `2-step` + sparse attention | ~120-145 s | faces and expressions match the draft; promote-level sharpness |
| 2K, `2-step` + sparse attention | ~215 s | also rebuilds faces only ~25 px tall in the draft |
| 1080p, NVIDIA's `up3` (dense) | ~210 s | |

What it does **not** do: invent texture. A generative refiner such as an LTX detail pass gives more
micro-detail (skin, stone, water) but can change small faces; this keeps them. 2K scaled to 1080p is
no sharper than a 1080p run -- use 2K where faces are tiny.

## Nodes

| node | |
|---|---|
| **Load Open-H3 IC LoRA** | applies the IC LoRA **as NVIDIA publishes it** (diffusers layout). q/k/v and the SwiGLU halves are mapped onto ComfyUI's fused weights as slice patches; no converted file |
| **Open-H3 IC Conditioning** | draft frames (+ optional first-stage latent) -> positive conditioning, the starting latent (draft through the latent upscaler, audio held silent), and the real frame count |
| **Open-H3 IC Sigmas** | `2-step` (default), `up3`, `up4`, `hf8` from HyperFlow's 8-step grid |
| **MiniMax-H3 Chunked FFN** | runs the MLP over the sequence in chunks (exact). Needed for 2K on 32 GB; patches only this model |
| **Open-H3 IC Trim** | drops the padded tail |

`2-step` keeps `up3`'s starting noise (sigma 0.835) and drops the middle point. HyperFlow conditions each
step on the interval it jumps, and in our tests it gave the same faces and equal sharpness ~1.5x faster.

## Install

1. Clone into `ComfyUI/custom_nodes/`, plus [ComfyUI-Hyperflow](https://github.com/Saganaki22/ComfyUI-Hyperflow).
2. Models (none are shipped here):

| folder | file | from |
|---|---|---|
| `diffusion_models/` | `minimax_h3_ref2va_int8_convrot.safetensors` (**not** the `_pruned` one) | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `text_encoders/` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | Comfy-Org/MiniMax-H3 |
| `vae/` | `minimax_h3_video_vae_int8_convrot.safetensors` (or fp16), `minimax_h3_audio_vae_fp32.safetensors` | Comfy-Org/MiniMax-H3 |
| `latent_upscale_models/` | `minimax_h3_latent_upscaler_3d_bf16.safetensors` (= NVIDIA's `minimax_h3_latent_upscaler_3d_conv_v1_bf16`) | Comfy-Org/MiniMax-H3 or the IC repo |
| `loras/` | `open_h3_regenerate_2k_ic_lora.safetensors` | [the IC repo](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/tree/main/lora) |
| `hyperflow/` | `custom_node_hyperflow_8step_v1.0_comfyui.safetensors` (the **full** build) | [drbaph/Hyperflow-Comfyui](https://huggingface.co/drbaph/Hyperflow-Comfyui) |

**Why the non-pruned base:** HyperFlow's two-time (t, r) conditioning needs the time embedder the pruned
checkpoints drop. On a pruned base HyperFlow runs backbone-only and the output is ~25% softer; its
experimental curve refit only covers the full 8-step grid, so it cannot run `2-step`.

## Workflow

`example_workflows/openh3_ic_1080p_api.json` (API format):

    Load Diffusion Model (ref2va, non-pruned) -> Apply HyperFlow (bypass) -> Load Open-H3 IC LoRA
      -> MiniMax-H3 Chunked FFN -> Model Sparse Attention (sol-attn, tau 1.0, dense blocks 0,1, sink exact_kv)
    Load Video -> Get Video Components -> Open-H3 IC Conditioning -> SamplerCustomAdvanced (euler, CFG 1,
      sigmas from Open-H3 IC Sigmas fed by Apply HyperFlow's SIGMAS) -> VAE Decode -> Open-H3 IC Trim

- `sink_conditioning: exact_kv` keeps attention to the draft exact and sparsifies only target-to-target;
  tau above 1.0 saved little and softened faces in our tests.
- Pass the draft's own first-stage latent as `draft_latent` when you have it: re-encoding the decoded
  MP4 moves the latent by ~30% RMS, and the kept latent gave a ~30% sharper result with the same faces.
- **Longer clips:** the LoRA was trained on 124 frames. Split into overlapping 124-frame windows and
  crossfade the overlaps; in our test a 15 s, three-shot take spliced with no visible join.

## Differences from NVIDIA's reference runtime

int8 base instead of bf16; LMS Sharp off (its curve branches target pruned bases); silence encoded by
ComfyUI's audio VAE rather than NVIDIA's stored latent; ComfyUI's own Ref2VA presentation of `<Video 1>`.
Against a run reproducing NVIDIA's pieces more literally, output matched at ~34 dB with the same faces.

## Licence

The code here is Apache-2.0. **The models are not**: MiniMax-H3 and its derivatives (HyperFlow, the IC
LoRA) are under the MiniMax H3 Community License, which among other terms excludes some territories.
Check it before use. This repository ships no weights.

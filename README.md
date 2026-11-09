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

All nodes are in the `openh3_ic` category. Hover any socket in ComfyUI for the same notes.

### Load Open-H3 IC LoRA

Applies NVIDIA's IC LoRA **as published** (diffusers layout). q/k/v and the two SwiGLU halves are mapped
onto ComfyUI's fused weights as slice patches, so there is no converted file.

| | name | type | |
|---|---|---|---|
| in | `model` | MODEL | MiniMax-H3 Ref2VA, **non-pruned**, with Apply HyperFlow already applied |
| in | `lora_name` | combo | `open_h3_regenerate_2k_ic_lora.safetensors` in `models/loras` |
| in | `strength` | FLOAT | 1.0 = the released model; other values untested |
| out | `MODEL` | MODEL | the patched model (312 LoRA modules -> 364 patches) |

### MiniMax-H3 Chunked FFN

Runs the MLP over the token sequence in chunks. Exact, because the MLP is applied to each token on its
own; patches only this model. Needed for 2K on a 32 GB card (the draft in context makes ~150k tokens),
harmless at 1080p.

| | name | type | |
|---|---|---|---|
| in | `model` | MODEL | any MiniMax-H3 model |
| in | `chunk_tokens` | INT | tokens per MLP pass (default 24576). Lower = less VRAM, marginally slower |
| out | `MODEL` | MODEL | the same model with chunked MLPs |

### Open-H3 IC Conditioning

Puts the draft in context as `<Video 1>` with NVIDIA's task prompt, and builds the starting latent: the
draft latent through the 3D latent upscaler to the target size, with the audio held as clean silence.

| | name | type | |
|---|---|---|---|
| in | `clip` | CLIP | MiniMax-H3 text encoder (`qwen3vl_32b_minimax_h3_nvfp4_awq`, type `minimax`); reads the draft's frames |
| in | `vae` | VAE | MiniMax-H3 video VAE; encodes the draft, and decodes the result afterwards |
| in | `audio_vae` | VAE | MiniMax-H3 audio VAE; only encodes the silent track |
| in | `upscale_model` | LATENT_UPSCALE_MODEL | `minimax_h3_latent_upscaler_3d_bf16` from Load Latent Upscale Model |
| in | `draft` | IMAGE | the draft's frames at any frame rate. Picked at 24 fps with NVIDIA's rule, cover-resized and center-cropped to 1344x768 (768x1344 for a portrait target, untested); 113-124 frames at 24 fps |
| in | `source_fps` | FLOAT | the draft's frame rate; connect `fps` from Get Video Components |
| in | `width`, `height` | INT | output size: 1920x1088 (1080p), 2560x1440 (2K), 3840x2176 (4K; two upscale passes) |
| in | `caption` | STRING | optional: the draft's original prompt, appended as the scene description |
| in | `draft_latent` | LATENT, optional | the draft's own 1344x768 first-stage latent; replaces re-encoding the frames (sharper) |
| in | `allow_short` | BOOLEAN, optional | pad drafts under 113 frames with their last frame |
| in | `audio` | AUDIO, optional | the draft's audio; passed through fitted to the result. The model never hears it |
| out | `positive` | CONDITIONING | to a Basic Guider (CFG 1, no negative) |
| out | `LATENT` | LATENT | to SamplerCustomAdvanced `latent_image`; carries a noise mask that keeps the audio silent |
| out | `frames` | INT | the draft's real length at 24 fps, for Open-H3 IC Trim |
| out | `audio` | AUDIO | the draft's audio trimmed or padded to the result, for Create Video |

### Open-H3 IC Sigmas

| | name | type | |
|---|---|---|---|
| in | `hyperflow_sigmas` | SIGMAS | the SIGMAS output of Apply HyperFlow (its 9-point grid) |
| in | `schedule` | combo | `2-step` (default): 0.835 -> 0.469 -> 0, fastest, same faces and sharpness as `up3` in our tests. `up3`: NVIDIA's default. `up4`: from 0.923. `hf8`: all 8 steps from noise |
| out | `SIGMAS` | SIGMAS | to SamplerCustomAdvanced, with the `euler` sampler |

### Open-H3 IC Trim

| | name | type | |
|---|---|---|---|
| in | `images` | IMAGE | the decoded result (124 frames) |
| in | `frames` | INT | connect `frames` from Open-H3 IC Conditioning |
| out | `IMAGE` | IMAGE | the result at the draft's own length |

### Third-party node used

**Apply HyperFlow (MiniMax-H3 8-Step)** comes from [ComfyUI-Hyperflow](https://github.com/Saganaki22/ComfyUI-Hyperflow)
(Apache-2.0). The IC LoRA was trained on top of HyperFlow, whose two-time (t, r) step conditioning
needs the non-pruned base. Use `lora_mode: bypass`; its SIGMAS output feeds Open-H3 IC Sigmas.

**Do not load HyperFlow as a plain LoRA.** That applies its backbone weights without the two-time
conditioning. Measured on the same draft, seed and `up3` schedule: same faces, but **~24% softer**
(face Laplacian variance 244 against 323) and visibly mushier skin and stone.

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

## Workflows

| file | output | time on a 5090 (124 frames) |
|---|---|---|
| `example_workflows/openh3_ic_1080p_api.json` | 1920x1088 | ~120-150 s |
| `example_workflows/openh3_ic_2k_api.json` | 2560x1440 | ~215 s |

Both are API format (drag onto the canvas, or queue them over the API); the draft is `input/draft.mp4`.
They differ only in the conditioning node's width and height. The graph:

    Load Diffusion Model (ref2va, non-pruned) -> Apply HyperFlow (bypass) -> Load Open-H3 IC LoRA
      -> MiniMax-H3 Chunked FFN -> Model Sparse Attention (sol-attn, tau 1.0, dense blocks 0,1, sink exact_kv)
    Load Video -> Get Video Components (frames, fps, audio) -> Open-H3 IC Conditioning -> SamplerCustomAdvanced (euler, CFG 1,
      sigmas from Open-H3 IC Sigmas fed by Apply HyperFlow's SIGMAS) -> VAE Decode -> Open-H3 IC Trim

- `sink_conditioning: exact_kv` keeps attention to the draft exact and sparsifies only target-to-target;
  tau above 1.0 saved little and softened faces in our tests.
- Pass the draft's own first-stage latent as `draft_latent` when you have it: re-encoding the decoded
  MP4 moves the latent by ~30% RMS, and the kept latent gave a ~30% sharper result with the same faces.
- **Longer clips:** the LoRA was trained on 124 frames. Split into overlapping 124-frame windows and
  crossfade the overlaps; in our test a 15 s, three-shot take spliced with no visible join.

## Differences from NVIDIA's reference runtime

- int8 base instead of bf16, and LMS Sharp off (its 50 curve branches target pruned bases).
- The silent audio track is encoded with ComfyUI's audio VAE instead of NVIDIA's stored latent. Measured:
  identical to 0.001 RMS over the first 205 of 207 audio frames; only the encoder's last two edge frames
  differ, and those rows are pinned, never generated.
- `<Video 1>` is presented to the text encoder by ComfyUI's own Ref2VA node.
- Draft preparation follows NVIDIA's `read_draft` (24 fps index rule, cover crop, 113-frame minimum).
- The IC LoRA's patches equal NVIDIA's `B @ A` deltas exactly (checked on CPU against the fused
  q/k/v and SwiGLU weights).

## Licence

The code here is Apache-2.0. **The models are not**: MiniMax-H3 and its derivatives (HyperFlow, the IC
LoRA) are under the MiniMax H3 Community License, which among other terms excludes some territories.
Check it before use. This repository ships no weights.

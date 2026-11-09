"""The draft in context: conditioning and the starting latent for Open-H3-Regenerate-2K-IC.

Mirrors NVIDIA's reference runtime (infer.py / ref2va_ic in the model repo):
  * the draft is <Video 1> of a Ref2VA request, prompt = TASK + "\n\nScene description: " + caption;
    the reference rows carry the draft latent at timestep 0.999, as ComfyUI's own Ref2VA does;
  * the target starts from the draft latent through the 3D latent upscaler, noised by the sampler;
  * the target audio is clean silence held fixed (noise mask 0), never generated.
Everything H3-specific is done by ComfyUI's own MiniMax-H3 nodes, called in process, so this
follows core when core changes.
"""
import torch

import comfy.nested_tensor
from comfy_extras import nodes_minimax_h3 as h3

TASK = ("Enhance <Video 1> into a high-quality version of the same video. "
        "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
        "Restore fine details and remove degradation. Do not continue or change the scene.")
FRAMES = 124            # the clip length the IC LoRA was trained on (5.2 s at 24 fps)
MIN_FRAMES = 113        # NVIDIA refuses shorter drafts; the tail is padded with the last frame
UPSCALE_STAGE = 1440    # targets above 1440 rows are reached in two latent-upscale passes, as NVIDIA does


def _result(output):
    return output.result if hasattr(output, "result") else output


def pad_frames(frames, allow_short=False):
    n = frames.shape[0]
    if n > FRAMES:
        raise ValueError(f"the draft has {n} frames; Open-H3-Regenerate-2K-IC takes at most {FRAMES} "
                         "(5.2 s at 24 fps). Split longer clips into overlapping windows.")
    if n < MIN_FRAMES and not allow_short:
        raise ValueError(f"the draft has {n} frames; the model needs at least {MIN_FRAMES}. "
                         "Enable allow_short to pad it with its last frame anyway.")
    if n < FRAMES:
        frames = torch.cat([frames, frames[-1:].expand(FRAMES - n, *frames.shape[1:])], 0)
    return frames, n


def _upscale(upscaler, latent, width, height):
    node = h3.MiniMaxH3LatentUpscaleWithModel
    if min(width, height) > UPSCALE_STAGE:
        scale = UPSCALE_STAGE / min(width, height)
        mid_w, mid_h = round(width * scale / 32) * 32, round(height * scale / 32) * 32
        latent = _result(node.execute(upscaler, {"samples": latent}, mid_w, mid_h))[0]["samples"]
    return _result(node.execute(upscaler, {"samples": latent}, width, height))[0]["samples"]


def _silence(audio_vae, audio_t):
    rate = getattr(audio_vae, "audio_sample_rate", 32000)
    samples = round(FRAMES / 24 * rate) + rate // 10
    z = audio_vae.encode(torch.zeros(1, samples, 2))          # [1, 32, 2, T]
    if z.shape[-1] < audio_t:
        z = torch.cat([z, z[..., -1:].expand(*z.shape[:-1], audio_t - z.shape[-1])], -1)
    return z[..., :audio_t].float()


def build(clip, vae, audio_vae, upscaler, frames, width, height, caption="", draft_latent=None, allow_short=False):
    frames, real = pad_frames(frames, allow_short)
    prompt = TASK + "\n\nScene description: " + (caption or "")
    positive, empty = _result(h3.MiniMaxH3ReferenceToVideo.execute(
        clip=clip, prompt=prompt, width=width, height=height, length=FRAMES, vae=vae,
        ref_videos={"ref_video_0": frames}))
    refs = positive[0][1]["minimax_refs"]
    draft = refs[0]["latent"]
    if draft_latent is not None:
        kept = draft_latent["samples"]
        kept = kept.unbind()[0] if getattr(kept, "is_nested", False) else kept
        kept = kept[:, :, :draft.shape[2]].to(draft.dtype)
        if kept.shape != draft.shape:
            raise ValueError(f"draft_latent {tuple(kept.shape)} does not match the draft {tuple(draft.shape)}: "
                             "it must be the 1344x768 first stage of this same clip")
        positive = [[c, dict(o, minimax_refs=[dict(refs[0], latent=kept)] + list(refs[1:]))] for c, o in positive]
        draft = kept
    video = _upscale(upscaler, draft.float(), width, height).float()
    audio_t = empty["samples"].unbind()[1].shape[-1]
    audio = _silence(audio_vae, audio_t)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio)),
              "noise_mask": comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(audio)))}
    return positive, latent, real

"""Draft preparation, as NVIDIA's reference runtime does it (ref2va_ic/media.py: read_draft).

* frames are picked at 24 fps from any source rate: output frame i is source frame int(i * fps / 24)
* each frame is cover-resized (bilinear, antialiased when shrinking) and center-cropped to the LQ
  canvas the IC LoRA was trained on, 1344x768 (768x1344 for a portrait target: untested)
* at least 113 real frames; the tail up to 124 repeats the last real frame

Pure torch, no ComfyUI imports, so it can be tested on its own.
"""
import math

import torch
import torch.nn.functional as F

FRAMES = 124
MIN_FRAMES = 113
# The MiniMax-H3 video VAE encodes in independent 17-frame clips, 5 latent frames each (causal,
# 4x temporal: 1 + 4), and drops the last 3 latents of the whole clip: 17k+5 frames -> 5k+2.
# Each clip starts its causal padding afresh, so latent block j depends on pixel frames
# 17j..17j+16 alone, and a window that starts on a block is an exact slice of the whole latent.
VAE_BLOCK_FRAMES = 17
VAE_BLOCK_LATENTS = 5
FPS = 24.0
CANVAS = (1344, 768)


def canvas_for(width, height):
    return CANVAS if width >= height else CANVAS[::-1]


def frames_at_24(source_count, source_fps):
    """How many 24 fps frames a source of `source_count` frames at `source_fps` covers."""
    if source_count < 1:
        return 0
    return int(math.floor((source_count - 1) * FPS / source_fps + 1e-9)) + 1


def pick_indices(source_count, source_fps, count=FRAMES):
    """NVIDIA's rule, then clamped: (source indices for `count` output frames, how many are real)."""
    wanted = [int(i * source_fps / FPS) for i in range(count)]
    real = sum(1 for w in wanted if w < source_count)
    return [min(w, source_count - 1) for w in wanted], real


def cover(frames, width, height):
    """[T, H, W, C] in [0, 1] -> [T, height, width, 3]: cover resize + center crop."""
    x = frames[..., :3].movedim(-1, 1).float()
    h, w = x.shape[-2:]
    scale = max(height / h, width / w)
    sh, sw = max(height, round(h * scale)), max(width, round(w * scale))
    if (sh, sw) != (h, w):
        x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False, antialias=scale < 1)
    top, left = (sh - height) // 2, (sw - width) // 2
    return x[..., top:top + height, left:left + width].movedim(1, -1).clamp(0, 1).contiguous()


def prepare(frames, source_fps, target_width, target_height, allow_short=False):
    """-> (frames [124, h, w, 3] on the LQ canvas, real frame count)."""
    n = int(frames.shape[0])
    total = frames_at_24(n, source_fps)
    if total > FRAMES:
        raise ValueError(f"the draft is {total} frames at 24 fps; Open-H3-Regenerate-2K-IC takes at most {FRAMES} "
                         "(5.2 s). Split longer clips into overlapping windows.")
    indices, real = pick_indices(n, source_fps)
    if real < MIN_FRAMES and not allow_short:
        raise ValueError(f"the draft gives {real} frames at 24 fps; the model needs at least {MIN_FRAMES} "
                         "(about 4.7 s). Enable allow_short to pad it with its last frame anyway.")
    picked = frames.index_select(0, torch.tensor(indices, device=frames.device))
    width, height = canvas_for(target_width, target_height)
    return cover(picked, width, height), real


def latent_frames(pixel_frames):
    """Latent frames the H3 video VAE gives for `pixel_frames` (17k+5 -> 5k+2)."""
    blocks = -(-max(int(pixel_frames), 1) // VAE_BLOCK_FRAMES)
    return max(blocks * VAE_BLOCK_LATENTS - 3, 2)


def latent_window(kept_frames, offset, length):
    """(start, end) latent indices of the window that starts at pixel frame `offset`.

    `length` is the draft's latent length (37 for the 124-frame canvas). The offset must sit on
    a VAE block, or no slice of the kept latent is the window's latent. Raises when the kept
    latent ends before the window does.
    """
    offset = int(offset)
    if offset < 0 or offset % VAE_BLOCK_FRAMES:
        raise ValueError(f"latent_offset {offset} must be a multiple of {VAE_BLOCK_FRAMES} pixel frames: "
                         "the H3 VAE encodes in 17-frame blocks, so only a window starting on one is a "
                         "slice of the kept latent")
    start = offset // VAE_BLOCK_FRAMES * VAE_BLOCK_LATENTS
    end = start + int(length)
    if end > int(kept_frames):
        raise ValueError(f"draft_latent has {kept_frames} latent frames; the window at pixel frame {offset} "
                         f"needs {end} (it must cover pixel frames {offset}..{offset + FRAMES - 1})")
    return start, end


def kept_blocks(kept_frames, length):
    """How many latent frames of a kept latent SHORTER than the draft stand in for the draft's own.

    Only whole VAE blocks: a kept first stage of 17n+5 frames is 5n+2 latents, and its last 2 are
    a partial block whose other 3 latents would come from a different encode. Blocks are encoded
    independently, so the draft's own encode of blocks n.. (its real tail, then the last frame
    repeated) follows on exactly. Raises when the kept latent holds no whole block, or is not
    shorter than the draft (the ordinary cut covers that).
    """
    kept_frames, length = int(kept_frames), int(length)
    if kept_frames >= length:
        raise ValueError(f"draft_latent has {kept_frames} latent frames; it is not shorter than the draft's {length}")
    whole = (kept_frames // VAE_BLOCK_LATENTS) * VAE_BLOCK_LATENTS
    if whole < VAE_BLOCK_LATENTS:
        raise ValueError(f"draft_latent has {kept_frames} latent frames, under one {VAE_BLOCK_FRAMES}-frame VAE "
                         "block; draft from the frames instead")
    return whole


def fit_audio(audio, real_frames):
    """Trim or zero-pad the draft's audio to the result's length (real_frames at 24 fps). None stays None."""
    if audio is None:
        return None
    wave, rate = audio["waveform"], int(audio["sample_rate"])
    want = round(real_frames / FPS * rate)
    if wave.shape[-1] >= want:
        wave = wave[..., :want]
    else:
        wave = torch.cat([wave, wave.new_zeros(*wave.shape[:-1], want - wave.shape[-1])], -1)
    return {"waveform": wave.contiguous(), "sample_rate": rate}

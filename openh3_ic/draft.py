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

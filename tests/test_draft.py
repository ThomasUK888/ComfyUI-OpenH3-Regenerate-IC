import pytest
import torch

from openh3_ic import draft


def test_nvidia_index_rule_at_30fps():
    indices, real = draft.pick_indices(150, 30.0)
    assert indices[:6] == [0, 1, 2, 3, 5, 6]          # int(i * 30 / 24)
    assert real == 120                                  # 150 frames at 30 fps cover 120 at 24 fps
    assert indices[-1] == 149                           # the tail is clamped to the last frame


def test_frames_at_24():
    assert draft.frames_at_24(124, 24.0) == 124
    assert draft.frames_at_24(150, 30.0) == 120
    assert draft.frames_at_24(1, 24.0) == 1


def test_cover_crops_to_the_canvas():
    frames = torch.rand(3, 1080, 1920, 3)
    out = draft.cover(frames, 1344, 768)
    assert out.shape == (3, 768, 1344, 3)
    square = draft.cover(torch.rand(2, 1000, 1000, 3), 1344, 768)
    assert square.shape == (2, 768, 1344, 3)            # cropped top and bottom, not stretched


def test_canvas_follows_the_target_orientation():
    assert draft.canvas_for(2560, 1440) == (1344, 768)
    assert draft.canvas_for(1088, 1920) == (768, 1344)


def test_prepare_pads_and_reports_the_real_length():
    frames = torch.rand(118, 72, 128, 3)
    out, real = draft.prepare(frames, 24.0, 1920, 1088)
    assert out.shape == (124, 768, 1344, 3) and real == 118
    assert torch.equal(out[-1], out[117])               # padded with the last real frame


def test_prepare_refuses_short_and_long_drafts():
    with pytest.raises(ValueError, match="at least 113"):
        draft.prepare(torch.rand(100, 72, 128, 3), 24.0, 1920, 1088)
    out, real = draft.prepare(torch.rand(100, 72, 128, 3), 24.0, 1920, 1088, allow_short=True)
    assert real == 100 and out.shape[0] == 124
    with pytest.raises(ValueError, match="at most 124"):
        draft.prepare(torch.rand(200, 72, 128, 3), 24.0, 1920, 1088)


def test_fit_audio_trims_and_pads():
    rate = 48000
    long = {"waveform": torch.ones(1, 2, rate * 10), "sample_rate": rate}
    assert draft.fit_audio(long, 120)["waveform"].shape[-1] == rate * 5
    short = {"waveform": torch.ones(1, 2, rate), "sample_rate": rate}
    fitted = draft.fit_audio(short, 48)["waveform"]
    assert fitted.shape[-1] == rate * 2 and fitted[..., rate:].abs().sum() == 0
    assert draft.fit_audio(None, 120) is None

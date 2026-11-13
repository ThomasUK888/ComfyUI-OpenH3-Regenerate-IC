import pytest

from openh3_ic import draft


def test_latent_frames_follow_the_17k_plus_5_grid():
    assert draft.latent_frames(5) == 2
    assert draft.latent_frames(124) == 37          # the canvas
    assert draft.latent_frames(141) == 42
    assert draft.latent_frames(362) == 107


def test_offset_zero_is_the_old_cut():
    assert draft.latent_window(42, 0, 37) == (0, 37)


def test_a_block_offset_moves_five_latents_per_17_frames():
    assert draft.latent_window(107, 17, 37) == (5, 42)
    assert draft.latent_window(107, 85, 37) == (25, 62)


def test_the_last_window_that_fits():
    # A 362-frame first stage (107 latents) holds a 124-frame window up to frame 238 = 14 * 17.
    assert draft.latent_window(107, 238, 37) == (70, 107)
    with pytest.raises(ValueError, match="needs 112"):
        draft.latent_window(107, 255, 37)


@pytest.mark.parametrize("offset", [1, 16, 24, 100, -17])
def test_an_offset_off_the_block_grid_is_refused(offset):
    with pytest.raises(ValueError, match="multiple of 17"):
        draft.latent_window(107, offset, 37)


def test_a_window_is_the_same_blocks_as_its_own_encode():
    """Pixel frame 17j starts latent block j: a window [17j, 17j + 124) of a clip is latent frames
    [5j, 5j + 37), and its own encode is 37 latents long."""
    for j in range(8):
        start, end = draft.latent_window(draft.latent_frames(17 * j + 124), 17 * j, draft.latent_frames(124))
        assert (start, end) == (5 * j, 5 * j + 37)

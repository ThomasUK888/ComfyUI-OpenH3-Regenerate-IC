"""The samplers of Open-H3-Regenerate-2K-IC, as slices of HyperFlow's 8-step grid.

`up3` / `up4` are NVIDIA's (the last 3 / 4 grid points, starting from the latent-upscaled draft);
`hf8` starts from noise. `2-step` keeps up3's starting noise level and drops the middle point:
measured on a 5090 it is ~1.5x faster with equal sharpness and the same faces (HyperFlow is a
flow map, so a step is conditioned on the interval it jumps). Pure Python for testing.
"""
SCHEDULES = ("2-step", "up3", "up4", "hf8")


def pick(grid, schedule):
    """grid: the 9 HyperFlow sigmas (1.0 ... 0.0) -> the sigmas to sample, and whether to start from noise."""
    g = [float(s) for s in grid]
    if len(g) != 9 or g[-1] != 0.0:
        raise ValueError(f"expected HyperFlow's 9-point 8-step grid, got {len(g)} points")
    if schedule == "hf8":
        return g, True
    if schedule == "up4":
        return g[4:], False
    if schedule == "up3":
        return g[5:], False
    if schedule == "2-step":
        return [g[5], g[7], g[8]], False
    raise ValueError(f"unknown schedule {schedule!r}; one of {SCHEDULES}")

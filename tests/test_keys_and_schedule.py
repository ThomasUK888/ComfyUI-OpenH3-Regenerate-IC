import numpy as np
import pytest

from openh3_ic import keys, schedule

Q, F = 10, 12         # one of q/k/v, one SwiGLU half -- deliberately NOT the hidden size (8)
HIDDEN = 8


def test_qkv_go_to_their_third_of_the_fused_projection():
    for part, letter in enumerate("qkv"):
        [(suffix, (key, (dim, off, size)))] = keys.target_for(f"transformer_blocks.7.attn.to_{letter}", Q, F)
        assert (suffix, key, dim, off, size) == ("", "diffusion_model.blocks.7.attn.qkv_proj.weight", 0, part * Q, Q)


def test_token_refiner_blocks_are_renamed():
    [(_, target)] = keys.target_for("token_refiner.refiner_blocks.1.attn.to_out.0", Q, F)
    assert target == "diffusion_model.token_refiner.blocks.1.attn.out_proj.weight"


def test_ffn_halves_swap():
    targets = dict(keys.target_for("transformer_blocks.0.ff.net.0.proj", Q, F))
    assert targets[keys.VALUE_HALF] == ("diffusion_model.blocks.0.mlp.fc1.weight", (0, F, F))
    assert targets[keys.GATE_HALF] == ("diffusion_model.blocks.0.mlp.fc1.weight", (0, 0, F))
    assert keys.target_for("transformer_blocks.0.ff.net.2", Q, F) == [("", "diffusion_model.blocks.0.mlp.fc2.weight")]


def test_unknown_modules_are_refused():
    for bad in ("transformer_blocks.0.norm1", "time_embedder.linear_1", "blocks.0.attn.to_q"):
        with pytest.raises(KeyError):
            keys.target_for(bad, Q, F)


def test_modules_requires_both_factors():
    k = ["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"]
    with pytest.raises(ValueError):
        keys.modules(k)
    assert keys.modules(k + ["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"]) == ["transformer_blocks.0.attn.to_q"]


def _comfy_delta(module_deltas):
    """Lay diffusers per-module deltas into ComfyUI's fused layout via target_for, as ComfyUI's slice patches do."""
    out = {}
    for module, delta in module_deltas.items():
        for suffix, target in keys.target_for(module, Q, F):
            part = delta[:F] if suffix == keys.VALUE_HALF else delta[F:] if suffix == keys.GATE_HALF else delta
            key, sl = (target, None) if isinstance(target, str) else target
            rows = {"qkv_proj": 3 * Q, "fc1": 2 * F}.get(key.split(".")[-2], part.shape[0])
            w = out.setdefault(key, np.zeros((rows, part.shape[1])))
            if sl is None:
                w += part
            else:
                _, off, size = sl
                w[off:off + size] += part
    return out


def test_dense_layout_matches_comfyui_order():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(Q, HIDDEN)) for _ in range(3))
    value, gate = rng.normal(size=(F, HIDDEN)), rng.normal(size=(F, HIDDEN))
    out = _comfy_delta({"transformer_blocks.0.attn.to_q": q, "transformer_blocks.0.attn.to_k": k,
                        "transformer_blocks.0.attn.to_v": v,
                        "transformer_blocks.0.ff.net.0.proj": np.concatenate([value, gate])})
    np.testing.assert_array_equal(out["diffusion_model.blocks.0.attn.qkv_proj.weight"], np.concatenate([q, k, v]))
    np.testing.assert_array_equal(out["diffusion_model.blocks.0.mlp.fc1.weight"], np.concatenate([gate, value]))


GRID = [1.0, 0.99391, 0.984288, 0.966064, 0.923077, 0.834942, 0.696852, 0.468753, 0.0]   # NVIDIA config.json


def test_schedules_are_nvidias():
    assert schedule.pick(GRID, "up3") == ([0.834942, 0.696852, 0.468753, 0.0], False)
    assert schedule.pick(GRID, "up4")[0] == [0.923077, 0.834942, 0.696852, 0.468753, 0.0]
    assert schedule.pick(GRID, "hf8") == (GRID, True)
    assert schedule.pick(GRID, "2-step") == ([0.834942, 0.468753, 0.0], False)


def test_schedule_refuses_another_grid():
    with pytest.raises(ValueError):
        schedule.pick(GRID[:-1], "up3")
    with pytest.raises(ValueError):
        schedule.pick(GRID, "up5")

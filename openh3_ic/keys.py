"""NVIDIA's diffusers-layout IC LoRA keys -> ComfyUI MiniMax-H3 weights.

Pure Python (no torch) so it can be tested on its own. The LoRA targets 312 linears of
`transformer_ref`: q/k/v/out and the two MLP projections of 2 token-refiner and 50 DiT blocks.
ComfyUI fuses q/k/v into one `attn.qkv_proj` and stores the SwiGLU input projection `mlp.fc1`
with its halves the other way round from diffusers' `ff.net.0.proj`:

    diffusers ff.net.0.proj rows = [value, gate]      ComfyUI mlp.fc1 rows = [gate, value]

So q, k and v each patch their own third of qkv_proj, and the two row halves of ff.net.0.proj
patch the opposite halves of fc1 -- expressed as ComfyUI weight slices `(key, (dim, offset, size))`
rather than by building block-diagonal or reordered matrices.
"""
import re

PREFIX = "transformer."
A, B = ".lora_A.weight", ".lora_B.weight"
VALUE_HALF, GATE_HALF = ".__value", ".__gate"   # synthetic LoRA names for the two halves of ff.net.0.proj

_BLOCK = re.compile(r"^(token_refiner\.refiner_blocks|transformer_blocks)\.(\d+)\.(.+)$")


def comfy_block(diffusers_block: str) -> str:
    kind, index = diffusers_block.rsplit(".", 1)
    return f"{'token_refiner.blocks' if kind.startswith('token_refiner') else 'blocks'}.{index}"


def target_for(module: str, hidden: int, ffn: int):
    """diffusers module (without `transformer.`) -> list of (lora_suffix, comfy target).

    A target is a diffusion_model weight key, or `(key, (dim, offset, size))` for a slice of it.
    `lora_suffix` is "" for the module's own LoRA, or one of the synthetic half names.
    """
    m = _BLOCK.match(module)
    if not m:
        raise KeyError(f"not an IC LoRA target: {module}")
    block, rest = comfy_block(f"{m.group(1)}.{m.group(2)}"), m.group(3)
    key = lambda name: f"diffusion_model.{block}.{name}.weight"
    if rest in ("attn.to_q", "attn.to_k", "attn.to_v"):
        part = "qkv".index(rest[-1])
        return [("", (key("attn.qkv_proj"), (0, part * hidden, hidden)))]
    if rest == "attn.to_out.0":
        return [("", key("attn.out_proj"))]
    if rest == "ff.net.0.proj":
        return [(VALUE_HALF, (key("mlp.fc1"), (0, ffn, ffn))),
                (GATE_HALF, (key("mlp.fc1"), (0, 0, ffn)))]
    if rest == "ff.net.2":
        return [("", key("mlp.fc2"))]
    raise KeyError(f"not an IC LoRA target: {module}")


def modules(state_keys):
    """The diffusers module names a LoRA state dict carries (both factors required)."""
    roots = {k[len(PREFIX):-len(A)] for k in state_keys if k.startswith(PREFIX) and k.endswith(A)}
    missing = [r for r in roots if PREFIX + r + B not in state_keys]
    if missing:
        raise ValueError(f"LoRA modules without a lora_B: {sorted(missing)[:3]}")
    return sorted(roots)

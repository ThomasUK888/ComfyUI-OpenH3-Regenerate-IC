"""Apply NVIDIA's IC LoRA, as published, to ComfyUI's MiniMax-H3 model -- no converted file."""
import logging

import comfy.lora
import comfy.utils

from . import keys

log = logging.getLogger("openh3_ic")


def _dims(model):
    dm = model.get_model_object("diffusion_model")
    hidden = int(dm.hidden_size)
    ffn = int(dm.blocks[0].mlp.fc2.in_features)
    return hidden, ffn


def patches_for(model, state):
    """IC LoRA state dict (diffusers layout) -> ComfyUI patch dict for `model`."""
    hidden, ffn = _dims(model)
    lora, to_load = {}, {}
    for module in keys.modules(state.keys()):
        a, b = state[keys.PREFIX + module + keys.A], state[keys.PREFIX + module + keys.B]
        for suffix, target in keys.target_for(module, hidden, ffn):
            name = keys.PREFIX + module + suffix
            if suffix == keys.VALUE_HALF:
                b_part = b[:ffn]
            elif suffix == keys.GATE_HALF:
                b_part = b[ffn:]
            else:
                b_part = b
            lora[name + keys.A], lora[name + keys.B] = a, b_part.contiguous()
            to_load[name] = target
    patches = comfy.lora.load_lora(lora, to_load, log_missing=True)
    if len(patches) != len(to_load):
        raise RuntimeError(f"only {len(patches)} of {len(to_load)} IC LoRA patches were built")
    return patches


def apply(model, path, strength):
    state = comfy.utils.load_torch_file(path, safe_load=True)
    modules = keys.modules(state.keys())
    if len(modules) != 312:
        log.warning("[openh3_ic] expected 312 LoRA modules (the released IC LoRA), found %d", len(modules))
    patches = patches_for(model, state)
    out = model.clone()
    applied = out.add_patches(patches, strength)
    if len(applied) != len(patches):
        missing = sorted(str(k) for k in set(patches) - set(applied))[:3]
        raise RuntimeError(f"{len(patches) - len(applied)} IC LoRA patches matched no model weight, e.g. {missing}. "
                           "Is this the MiniMax-H3 Ref2VA model?")
    log.info("[openh3_ic] IC LoRA: %d modules -> %d patches at strength %s", len(modules), len(patches), strength)
    return out

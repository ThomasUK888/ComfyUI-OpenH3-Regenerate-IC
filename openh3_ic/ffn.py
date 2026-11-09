"""Run the H3 MLP over the token sequence in chunks, on one model only.

The MLP is applied to every token independently, so splitting the rows is exact. At 2K a whole
sequence (~150k tokens with the draft in context) makes fc1's output alone ~9.5 GB, which does not
fit beside the weights on a 32 GB card. This is a per-model object patch rather than a global
monkeypatch, so other renders in the same ComfyUI are untouched.
"""
import torch


def _chunked(mlp, chunk):
    forward = type(mlp).forward

    def run(x):
        if x.shape[0] <= chunk:
            return forward(mlp, x)
        out = torch.empty(x.shape[:-1] + (mlp.fc2.out_features,), dtype=x.dtype, device=x.device)
        for i in range(0, x.shape[0], chunk):
            out[i:i + chunk] = forward(mlp, x[i:i + chunk])
        return out
    return run


def apply(model, chunk):
    out = model.clone()
    dm = out.get_model_object("diffusion_model")
    for i, block in enumerate(dm.blocks):
        out.add_object_patch(f"diffusion_model.blocks.{i}.mlp.forward", _chunked(block.mlp, chunk))
    return out

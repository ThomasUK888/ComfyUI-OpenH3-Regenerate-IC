import folder_paths
from comfy_api.latest import ComfyExtension, io

import torch

from . import conditioning, ffn, lora
from . import schedule as sched


class OpenH3ICLoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OpenH3ICLoraLoader", display_name="Load Open-H3 IC LoRA", category="openh3_ic",
            description="Applies NVIDIA's Open-H3-Regenerate-2K-IC LoRA (the released diffusers-layout file, "
                        "no conversion needed) to a MiniMax-H3 Ref2VA model. Chain after Apply HyperFlow.",
            inputs=[io.Model.Input("model"),
                    io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                    io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05)],
            outputs=[io.Model.Output()])

    @classmethod
    def execute(cls, model, lora_name, strength) -> io.NodeOutput:
        return io.NodeOutput(lora.apply(model, folder_paths.get_full_path_or_raise("loras", lora_name), strength))


class OpenH3ICChunkedFFN(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OpenH3ICChunkedFFN", display_name="MiniMax-H3 Chunked FFN", category="openh3_ic",
            description="Runs the H3 MLP over the sequence in chunks (exact). Needed for 2K on 32 GB cards; "
                        "patches this model only.",
            inputs=[io.Model.Input("model"),
                    io.Int.Input("chunk_tokens", default=24576, min=1024, max=1 << 20, step=1024)],
            outputs=[io.Model.Output()])

    @classmethod
    def execute(cls, model, chunk_tokens) -> io.NodeOutput:
        return io.NodeOutput(ffn.apply(model, chunk_tokens))


class OpenH3ICConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OpenH3ICConditioning", display_name="Open-H3 IC Conditioning", category="openh3_ic",
            description="Puts a low-resolution H3 draft (up to 124 frames at 24 fps) in context as <Video 1> and "
                        "builds the starting latent: the draft through the latent upscaler, audio held silent.",
            inputs=[io.Clip.Input("clip"), io.Vae.Input("vae"), io.Vae.Input("audio_vae"),
                    io.LatentUpscaleModel.Input("upscale_model"),
                    io.Image.Input("draft", tooltip="The draft's frames at 24 fps, at most 124."),
                    io.Int.Input("width", default=2560, min=256, max=4096, step=32),
                    io.Int.Input("height", default=1440, min=256, max=4096, step=32),
                    io.String.Input("caption", multiline=True, default="",
                                    tooltip="Optional: the prompt the draft was generated from."),
                    io.Latent.Input("draft_latent", optional=True,
                                    tooltip="Optional: the draft's own first-stage latent (1344x768). Sharper "
                                            "than re-encoding the decoded frames."),
                    io.Boolean.Input("allow_short", default=False, optional=True,
                                     tooltip="Pad drafts under 113 frames with their last frame.")],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(),
                     io.Int.Output(display_name="frames", tooltip="The draft's real frame count; trim the "
                                   "decoded video to it.")])

    @classmethod
    def execute(cls, clip, vae, audio_vae, upscale_model, draft, width, height, caption,
                draft_latent=None, allow_short=False) -> io.NodeOutput:
        positive, latent, real = conditioning.build(clip, vae, audio_vae, upscale_model, draft, width, height,
                                                    caption, draft_latent, allow_short)
        return io.NodeOutput(positive, latent, real)


class OpenH3ICSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OpenH3ICSigmas", display_name="Open-H3 IC Sigmas", category="openh3_ic",
            description="Picks the IC sampler from HyperFlow's 8-step grid. 2-step: up3's start, middle point "
                        "dropped (fastest, same quality in our tests). up3: NVIDIA's default.",
            inputs=[io.Sigmas.Input("hyperflow_sigmas", tooltip="The SIGMAS output of Apply HyperFlow."),
                    io.Combo.Input("schedule", options=list(sched.SCHEDULES), default="2-step")],
            outputs=[io.Sigmas.Output()])

    @classmethod
    def execute(cls, hyperflow_sigmas, schedule) -> io.NodeOutput:
        sigmas, _ = sched.pick(hyperflow_sigmas.tolist(), schedule)
        return io.NodeOutput(torch.tensor(sigmas, dtype=hyperflow_sigmas.dtype))


class OpenH3ICTrim(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OpenH3ICTrim", display_name="Open-H3 IC Trim", category="openh3_ic",
            description="Drops the padded tail: keeps the first `frames` images.",
            inputs=[io.Image.Input("images"), io.Int.Input("frames", default=124, min=1, max=100000)],
            outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, images, frames) -> io.NodeOutput:
        return io.NodeOutput(images[:frames])


class OpenH3ICExtension(ComfyExtension):
    async def get_node_list(self):
        return [OpenH3ICLoraLoader, OpenH3ICChunkedFFN, OpenH3ICConditioning, OpenH3ICSigmas, OpenH3ICTrim]

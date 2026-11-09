import folder_paths
import torch
from comfy_api.latest import ComfyExtension, io

from . import conditioning, ffn, lora
from . import schedule as sched


class OpenH3ICLoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OpenH3ICLoraLoader", display_name="Load Open-H3 IC LoRA", category="openh3_ic",
            description="Applies NVIDIA's Open-H3-Regenerate-2K-IC LoRA to a MiniMax-H3 Ref2VA model. Takes the file "
                        "exactly as NVIDIA publishes it (diffusers layout); nothing to convert. Chain it after "
                        "Apply HyperFlow, which the LoRA was trained on top of.",
            inputs=[
                io.Model.Input("model", tooltip="A MiniMax-H3 Ref2VA model with HyperFlow applied. Use the NON-pruned "
                                                "minimax_h3_ref2va checkpoint: HyperFlow's two-time conditioning needs "
                                                "the time embedder the pruned ones drop."),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"),
                               tooltip="open_h3_regenerate_2k_ic_lora.safetensors from Efficient-Large-Model/"
                                       "Open-H3-Regenerate-2K-IC (lora/), in models/loras."),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="1.0 is the released model. Other values are untested."),
            ],
            outputs=[io.Model.Output(tooltip="The model with the IC LoRA patched in (312 modules -> 364 patches: "
                                             "the SwiGLU input projection is two slices in ComfyUI's layout).")])

    @classmethod
    def execute(cls, model, lora_name, strength) -> io.NodeOutput:
        return io.NodeOutput(lora.apply(model, folder_paths.get_full_path_or_raise("loras", lora_name), strength))


class OpenH3ICChunkedFFN(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OpenH3ICChunkedFFN", display_name="MiniMax-H3 Chunked FFN", category="openh3_ic",
            description="Runs the H3 MLP over the token sequence in chunks. Exact (each token is independent), and "
                        "only this model is patched. Needed for 2K on a 32 GB card, where the draft in context makes "
                        "~150k tokens and one MLP activation alone is ~9.5 GB. Harmless at 1080p.",
            inputs=[
                io.Model.Input("model", tooltip="Any MiniMax-H3 model."),
                io.Int.Input("chunk_tokens", default=24576, min=1024, max=1 << 20, step=1024,
                             tooltip="Tokens per MLP pass. Lower uses less VRAM and is marginally slower; sequences "
                                     "shorter than this run in one pass."),
            ],
            outputs=[io.Model.Output(tooltip="The same model with its MLPs chunked.")])

    @classmethod
    def execute(cls, model, chunk_tokens) -> io.NodeOutput:
        return io.NodeOutput(ffn.apply(model, chunk_tokens))


class OpenH3ICConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OpenH3ICConditioning", display_name="Open-H3 IC Conditioning", category="openh3_ic",
            description="Puts the low-resolution draft in context as <Video 1> with NVIDIA's task prompt and builds "
                        "the starting latent: the draft latent through the 3D latent upscaler to the target size, with "
                        "the audio held as clean silence (never generated). Up to 124 frames at 24 fps per run.",
            inputs=[
                io.Clip.Input("clip", tooltip="The MiniMax-H3 text encoder (qwen3vl_32b_minimax_h3, type minimax). "
                                              "It reads the draft's frames as vision tokens."),
                io.Vae.Input("vae", tooltip="The MiniMax-H3 video VAE. Encodes the draft (unless draft_latent is "
                                            "given); use the same one to decode the result."),
                io.Vae.Input("audio_vae", tooltip="The MiniMax-H3 audio VAE. Only used to encode the silent audio "
                                                  "track the model expects beside the video."),
                io.LatentUpscaleModel.Input("upscale_model",
                                            tooltip="minimax_h3_latent_upscaler_3d_bf16 via Load Latent Upscale Model "
                                                    "(same weights as NVIDIA's latent_upscaler_3d_conv_v1)."),
                io.Image.Input("draft", tooltip="The draft's frames, 24 fps, 113-124 of them (see allow_short). Any "
                                                "size; it is fitted to the 1344x768 canvas the LoRA was trained on. "
                                                "16:9 is what has been tested."),
                io.Int.Input("width", default=2560, min=256, max=4096, step=32,
                             tooltip="Output width. 1920 (with height 1088) for 1080p, 2560 for 2K, 3840 for 4K."),
                io.Int.Input("height", default=1440, min=256, max=4096, step=32,
                             tooltip="Output height. 1088 for 1080p (crop to 1080 afterwards if needed), 1440 for 2K, "
                                     "2176 for 4K. Above 1440 the latent is upscaled in two passes, as NVIDIA does."),
                io.String.Input("caption", multiline=True, default="",
                                tooltip="Optional. The prompt the draft was generated from, unchanged; it is appended "
                                        "to NVIDIA's task prompt as the scene description. Empty works."),
                io.Latent.Input("draft_latent", optional=True,
                                tooltip="Optional. The draft's own first-stage latent (1344x768, before it was "
                                        "decoded), e.g. via Load Latent. Replaces re-encoding the frames: sharper "
                                        "results in our tests. Must be this same clip; it is cut to 124 frames."),
                io.Boolean.Input("allow_short", default=False, optional=True,
                                 tooltip="Allow drafts under 113 frames (NVIDIA's minimum). They are padded to 124 "
                                         "with their last frame; Open-H3 IC Trim removes the padding."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive",
                                       tooltip="Draft-in-context conditioning. Feed it to a Basic Guider (CFG 1, no "
                                               "negative)."),
                io.Latent.Output(tooltip="The starting latent at the target size, with a noise mask that keeps the "
                                         "audio silent. Feed it to SamplerCustomAdvanced's latent_image; the sampler "
                                         "adds the noise for the schedule's first sigma."),
                io.Int.Output(display_name="frames",
                              tooltip="The draft's real frame count. Connect to Open-H3 IC Trim to drop the padding."),
            ])

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
            description="Picks the IC sampler's sigmas from HyperFlow's 8-step grid.",
            inputs=[
                io.Sigmas.Input("hyperflow_sigmas", tooltip="The SIGMAS output of Apply HyperFlow (its 9-point grid)."),
                io.Combo.Input("schedule", options=list(sched.SCHEDULES), default="2-step",
                               tooltip="2-step: start at sigma 0.835, one jump to 0.469, then 0 -- fastest, same faces "
                                       "and sharpness as up3 in our tests. up3: NVIDIA's default (0.835, 0.697, 0.469, "
                                       "0). up4: from 0.923, a little more regeneration. hf8: all 8 steps from pure "
                                       "noise, slowest."),
            ],
            outputs=[io.Sigmas.Output(tooltip="Feed to SamplerCustomAdvanced with the euler sampler.")])

    @classmethod
    def execute(cls, hyperflow_sigmas, schedule) -> io.NodeOutput:
        sigmas, _ = sched.pick(hyperflow_sigmas.tolist(), schedule)
        return io.NodeOutput(torch.tensor(sigmas, dtype=hyperflow_sigmas.dtype))


class OpenH3ICTrim(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OpenH3ICTrim", display_name="Open-H3 IC Trim", category="openh3_ic",
            description="Keeps the first `frames` images: drops the tail a short draft was padded with.",
            inputs=[io.Image.Input("images", tooltip="The decoded result (124 frames)."),
                    io.Int.Input("frames", default=124, min=1, max=100000,
                                 tooltip="Connect the frames output of Open-H3 IC Conditioning.")],
            outputs=[io.Image.Output(tooltip="The result at the draft's own length.")])

    @classmethod
    def execute(cls, images, frames) -> io.NodeOutput:
        return io.NodeOutput(images[:frames])


class OpenH3ICExtension(ComfyExtension):
    async def get_node_list(self):
        return [OpenH3ICLoraLoader, OpenH3ICChunkedFFN, OpenH3ICConditioning, OpenH3ICSigmas, OpenH3ICTrim]

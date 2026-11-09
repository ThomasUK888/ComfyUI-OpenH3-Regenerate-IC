"""ComfyUI nodes for NVIDIA's Open-H3-Regenerate-2K-IC (MiniMax-H3 draft -> 1080p / 2K / 4K)."""
if __package__:   # pytest imports this file outside ComfyUI, where the nodes cannot load
    from .openh3_ic.nodes import OpenH3ICExtension

    async def comfy_entrypoint():
        return OpenH3ICExtension()

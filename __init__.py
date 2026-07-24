"""ComfyUI entry point for the 10MinVideoMaker custom-node package."""

from .tenminvideomaker.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

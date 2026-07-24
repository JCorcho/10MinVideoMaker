"""ComfyUI entry point for the 10MinVideoMaker custom-node package."""

from .tenminvideomaker.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .tenminvideomaker.server_api import register_routes

register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

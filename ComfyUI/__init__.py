"""ComfyUI custom-node entry point for FastVR."""

from __future__ import annotations

import sys
from pathlib import Path


# The extension is commonly symlinked from this repository into custom_nodes.
# Resolve the symlink so the sibling ``fastvr`` package remains importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


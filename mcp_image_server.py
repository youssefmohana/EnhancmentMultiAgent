#!/usr/bin/env python3
"""Legacy shim — use `src/enhancement_multiagent/mcp/server.py` instead."""
import warnings
warnings.warn("mcp_image_server.py is deprecated, use enhancement_multiagent.mcp.server", DeprecationWarning, stacklevel=2)
from enhancement_multiagent.mcp.server import *  # noqa: F401,F403
from enhancement_multiagent.mcp.server import mcp  # noqa: F401
if __name__ == "__main__":
    from enhancement_multiagent.mcp.server import mcp as _mcp
    _mcp.run(transport="stdio")

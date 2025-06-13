import sys
import logging

# Configure logging to stderr instead of stdout to avoid interfering with LSP JSON communication
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger("lean-lsp-mcp")

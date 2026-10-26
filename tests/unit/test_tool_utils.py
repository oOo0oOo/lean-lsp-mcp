from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from lean_lsp_mcp import tool_utils
from lean_lsp_mcp.utils import LeanToolError


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("timed out"),
        urllib.error.HTTPError("https://x", 503, "Service Unavailable", None, None),
        ValueError("unexpected character"),
    ],
)
async def test_urlopen_json_surfaces_request_errors(monkeypatch, error):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(tool_utils.urllib.request, "urlopen", fail)
    req = urllib.request.Request("https://leansearch.net/search")

    with pytest.raises(LeanToolError, match="Request to leansearch.net failed"):
        await tool_utils.urlopen_json(req, timeout=1)

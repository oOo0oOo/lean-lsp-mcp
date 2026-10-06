from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import lean_lsp_mcp
import pytest


def test_main_sets_security_env_flags(monkeypatch):
    for key in [
        "LEAN_LSP_MCP_ACTIVE_TRANSPORT",
        "LEAN_PROJECT_PATH",
        "LEAN_MCP_DISABLED_TOOLS",
        "LEAN_MCP_TOOL_DESCRIPTIONS",
        "LEAN_MCP_INSTRUCTIONS",
    ]:
        # Track these keys through monkeypatch so main() assignments are restored.
        monkeypatch.setenv(key, "__original__")

    captured: dict[str, str] = {}

    def fake_run(*, transport: str) -> None:
        captured["transport"] = transport

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lean-lsp-mcp",
            "--transport",
            "stdio",
            "--lean-project-path",
            "/tmp/project",
            "--disable-tools",
            "lean_run_code,lean_build",
            "--tool-descriptions",
            '{"lean_goal": "custom description"}',
            "--instructions",
            "You are a Lean expert.",
        ],
    )
    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", fake_run)
    # infer_project_path walks up looking for lean-toolchain; stub it out.
    monkeypatch.setattr(
        lean_lsp_mcp, "infer_project_path", lambda p, **kw: Path("/tmp/project")
    )

    lean_lsp_mcp.main()

    assert captured["transport"] == "stdio"
    assert lean_lsp_mcp.os.environ["LEAN_PROJECT_PATH"] == "/tmp/project"
    assert (
        lean_lsp_mcp.os.environ["LEAN_MCP_DISABLED_TOOLS"] == "lean_run_code,lean_build"
    )
    assert (
        lean_lsp_mcp.os.environ["LEAN_MCP_TOOL_DESCRIPTIONS"]
        == '{"lean_goal": "custom description"}'
    )
    assert lean_lsp_mcp.os.environ["LEAN_MCP_INSTRUCTIONS"] == "You are a Lean expert."


@pytest.mark.skipif(os.name != "posix", reason="POSIX shutdown signals")
@pytest.mark.parametrize(
    "transport,interrupt_error",
    [
        ("stdio", KeyboardInterrupt),
        ("stdio", asyncio.CancelledError),
        ("streamable-http", KeyboardInterrupt),
        ("sse", KeyboardInterrupt),
    ],
)
def test_main_sigterm_cancels_stdio_and_restores_handler(
    monkeypatch, transport, interrupt_error
):
    previous_handler = signal.getsignal(signal.SIGTERM)
    monkeypatch.setenv("LEAN_LSP_MCP_ACTIVE_TRANSPORT", "__original__")
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp", "--transport", transport])

    def interrupt(sig):
        assert sig == signal.SIGINT
        raise interrupt_error

    def run(**kwargs):
        assert kwargs["transport"] == transport
        handler = signal.getsignal(signal.SIGTERM)
        if transport == "stdio":
            assert callable(handler) and handler is not previous_handler
            handler(signal.SIGTERM, None)
        else:
            assert handler == previous_handler

    monkeypatch.setattr(signal, "raise_signal", interrupt)
    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", run)

    assert lean_lsp_mcp.main() == (143 if transport == "stdio" else 0)
    assert signal.getsignal(signal.SIGTERM) == previous_handler


def test_main_preserves_unrelated_cancellation(monkeypatch):
    previous_handler = signal.getsignal(signal.SIGTERM)
    monkeypatch.setenv("LEAN_LSP_MCP_ACTIVE_TRANSPORT", "__original__")
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp"])

    def run(**kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", run)
    with pytest.raises(asyncio.CancelledError):
        lean_lsp_mcp.main()
    assert signal.getsignal(signal.SIGTERM) == previous_handler

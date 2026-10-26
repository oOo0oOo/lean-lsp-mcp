from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import lean_lsp_mcp


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
    monkeypatch.setattr(
        lean_lsp_mcp, "create_server", lambda: SimpleNamespace(run=fake_run)
    )
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


def test_factory_observes_cli_overrides_before_server_construction(monkeypatch):
    import os

    monkeypatch.setenv("LEAN_MCP_INSTRUCTIONS", "environment instructions")
    monkeypatch.setenv("LEAN_MCP_DISABLED_TOOLS", "lean_goal")
    monkeypatch.setenv("LEAN_LSP_MCP_ACTIVE_TRANSPORT", "stdio")
    observed = {}

    def factory():
        observed.update(
            instructions=os.environ["LEAN_MCP_INSTRUCTIONS"],
            disabled=os.environ["LEAN_MCP_DISABLED_TOOLS"],
            transport=os.environ["LEAN_LSP_MCP_ACTIVE_TRANSPORT"],
        )
        return SimpleNamespace(run=lambda **kwargs: None)

    monkeypatch.setattr(lean_lsp_mcp, "create_server", factory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lean-lsp-mcp",
            "--instructions",
            "CLI instructions",
            "--disable-tools",
            "lean_build",
            "--transport",
            "streamable-http",
        ],
    )
    assert lean_lsp_mcp.main() == 0
    assert observed == {
        "instructions": "CLI instructions",
        "disabled": "lean_build",
        "transport": "streamable-http",
    }

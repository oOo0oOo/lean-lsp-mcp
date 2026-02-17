from __future__ import annotations

import sys

import lean_lsp_mcp


def test_main_sets_security_env_flags(monkeypatch):
    for key in [
        "LEAN_PROJECT_PATH",
        "LEAN_MCP_DISABLED_TOOLS",
        "LEAN_MCP_TOOL_DESCRIPTIONS",
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
            "--project-root",
            "/tmp/project",
            "--disable-tools",
            "lean_run_code,lean_build",
            "--tool-descriptions",
            '{"lean_goal": "custom description"}',
        ],
    )
    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", fake_run)

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

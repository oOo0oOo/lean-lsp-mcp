from __future__ import annotations

import sys
from pathlib import Path

import lean_lsp_mcp


def test_main_sets_security_env_flags(monkeypatch):
    for key in [
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

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, connect_stdio_client, result_json


@pytest.fixture()
def multi_attempt_false_success_file(test_project_path: Path) -> Path:
    path = test_project_path / "MultiAttemptFalseSuccess.lean"
    path.write_text(
        textwrap.dedent(
            """\
            import Mathlib

            def f (n : Nat) : Nat := n + 1
            def g (n : Nat) : Nat := f n + 2

            example : True := by
              let y := g 5
              suffices y + 3 = f 5 + 5 by trivial
              dsimp [g]
              omega
            """
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def repl_mcp_client_factory(
    repo_root: Path, test_project_path: Path
) -> Callable[[], AsyncContextManager[MCPClient]]:
    def find_repl_bin() -> Path | None:
        candidates = [
            test_project_path / ".lake" / "build" / "bin" / "repl",
            test_project_path
            / ".lake"
            / "packages"
            / "REPL"
            / ".lake"
            / "build"
            / "bin"
            / "repl",
        ]
        repl_path = next((p for p in candidates if p.exists()), None)
        if repl_path:
            return repl_path
        found = shutil.which("repl")
        return Path(found) if found else None

    repl_bin = find_repl_bin()
    if not repl_bin:
        lake_bin = repo_root / ".devenv" / "profile" / "bin" / "lake"
        lake_cmd = str(lake_bin) if lake_bin.exists() else "lake"
        try:
            subprocess.run([lake_cmd, "build", "repl"], cwd=test_project_path, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("REPL binary not found")
        repl_bin = find_repl_bin()
    if not repl_bin:
        pytest.skip("REPL binary not found")

    pythonpath_entries = [str(repo_root / "src")]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)

    env = {
        "PYTHONPATH": os.pathsep.join(pythonpath_entries),
        "LEAN_LOG_LEVEL": os.environ.get("LEAN_LOG_LEVEL", "ERROR"),
        "LEAN_LSP_TEST_MODE": "1",
        "LEAN_REPL": "true",
        "LEAN_REPL_PATH": str(repl_bin),
        "LEAN_REPL_TIMEOUT": "30",
    }

    def factory() -> AsyncContextManager[MCPClient]:
        return connect_stdio_client(
            sys.executable,
            ["-m", "lean_lsp_mcp", "--transport", "stdio"],
            env=env,
            cwd=repo_root,
        )

    return factory


@pytest.mark.xfail(
    strict=True,
    reason="REPL multi_attempt can report false success for let-bound locals",
)
@pytest.mark.asyncio
async def test_multi_attempt_repl_does_not_report_false_success(
    repl_mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    multi_attempt_false_success_file: Path,
) -> None:
    async with repl_mcp_client_factory() as client:
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(multi_attempt_false_success_file),
                "severity": "error",
            },
        )
        diagnostic_messages = [
            item["message"] for item in result_json(diagnostics)["result"]["items"]
        ]
        assert any("omega could not prove the goal" in msg for msg in diagnostic_messages)

        result = await client.call_tool(
            "lean_multi_attempt",
            {
                "file_path": str(multi_attempt_false_success_file),
                "line": 9,
                "snippets": ["dsimp [g]\nomega"],
            },
        )
        attempt = result_json(result)["items"][0]

        # A correct result should expose failure via remaining goals or diagnostics,
        # not silently report proof completion.
        assert attempt["goals"] or attempt["diagnostics"]

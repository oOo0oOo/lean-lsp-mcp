from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _transport_test_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(repo_root / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["LEAN_LOG_LEVEL"] = "INFO"
    env["LEAN_LSP_TEST_MODE"] = "1"
    return env


def test_stdio_transport_disconnect_exits_cleanly(repo_root: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "lean_lsp_mcp", "--transport", "stdio"],
        cwd=repo_root,
        env=_transport_test_env(repo_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    try:
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "transport-disconnect-test", "version": "1.0"},
            },
        }
        process.stdin.write(json.dumps(initialize_request) + "\n")
        process.stdin.flush()

        # Simulate abrupt client disconnect from the server write stream.
        process.stdout.close()
        process.stdin.close()

        try:
            returncode = process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait(timeout=5)
            raise AssertionError(
                "Server did not exit after transport disconnect."
            ) from exc

        stderr_output = process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert returncode == 0, stderr_output
    assert "Exception Group Traceback" not in stderr_output
    assert (
        "Exception ignored in: <_io.TextIOWrapper name='<stdout>'" not in stderr_output
    )
    assert "BrokenPipeError: [Errno 32] Broken pipe" not in stderr_output

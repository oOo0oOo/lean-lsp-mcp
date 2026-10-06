"""Exercise installer cancellation through the actual stdio CLI lifespan."""

import os
from pathlib import Path
import signal
import select
import subprocess
import sys
import textwrap
import time

import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signals")
@pytest.mark.parametrize(
    "shutdown_signal,ignore_sigint",
    [(signal.SIGINT, False), (signal.SIGTERM, False), (signal.SIGTERM, True)],
)
@pytest.mark.parametrize("hold_stdin_open", [False, True])
def test_cli_signal_during_initial_loogle_install(
    tmp_path, shutdown_signal, ignore_sigint, hold_stdin_open
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.30.0\n")
    (project / "lakefile.toml").write_text('name = "test"\n')
    pid_file = tmp_path / "installer.pid"
    installer = (
        "import os,time;from pathlib import Path;"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()));time.sleep(60)"
    )
    script = textwrap.dedent(f"""
        import signal, sys
        import lean_lsp_mcp
        from lean_lsp_mcp.loogle import LoogleManager

        if {ignore_sigint!r}:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        original_run = LoogleManager._run
        def controlled_run(self, cmd, timeout=300, cwd=None, env=None):
            return original_run(self, [sys.executable, '-c', {installer!r}],
                                timeout=15, cwd=cwd, env=env)
        LoogleManager._run = controlled_run
        LoogleManager._check_prerequisites = lambda self: (True, '')
        sys.argv = ['lean-lsp-mcp', '--loogle-local',
                    '--lean-project-path', {str(project)!r},
                    '--loogle-cache-dir', {str(tmp_path / "cache")!r}]
        raise SystemExit(lean_lsp_mcp.main())
    """)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env["LEAN_REPL"] = "false"
    installer_pid = None
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", script],
        env=env,
        stdin=subprocess.PIPE if hold_stdin_open else subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists():
            assert process.poll() is None, "CLI exited before starting the installer"
            assert time.monotonic() < deadline, "Installer did not start"
            time.sleep(0.01)
        installer_pid = int(pid_file.read_text())
        os.killpg(process.pid, shutdown_signal)
        # wait() keeps an open input pipe open, unlike communicate().
        process.wait(timeout=5)
        process.communicate(timeout=5)
        with pytest.raises(ProcessLookupError):
            os.kill(installer_pid, 0)
    finally:
        if installer_pid is None and pid_file.exists():
            installer_pid = int(pid_file.read_text())
        if installer_pid is not None:
            try:
                os.kill(installer_pid, 0)
            except ProcessLookupError:
                pass
            else:
                os.killpg(installer_pid, signal.SIGKILL)
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=3)


@pytest.mark.skipif(os.name != "posix", reason="POSIX shutdown signals")
@pytest.mark.parametrize("response_mode", ["small", "large", "blocked"])
@pytest.mark.timeout(15)
def test_cli_sigterm_after_initialize_with_stdin_open(response_mode):
    import json

    script = textwrap.dedent(f"""
        import sys, lean_lsp_mcp
        sys.argv = ['lean-lsp-mcp']
        if {response_mode != "small"!r}:
            sys.argv += ['--instructions', 'x' * (1024 * 1024)]
        raise SystemExit(lean_lsp_mcp.main())
    """)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env["LEAN_LOOGLE_LOCAL"] = "false"
    env["LEAN_REPL"] = "false"
    env.pop("LEAN_PROJECT_PATH", None)
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", script],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "shutdown-test", "version": "1"},
            },
        }
        process.stdin.write((json.dumps(request) + "\n").encode())
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 5)[0], "No initialize response"
        if response_mode != "blocked":
            response = json.loads(process.stdout.readline())
            assert response["id"] == 1 and "result" in response
            if response_mode == "large":
                assert response["result"]["instructions"] == "x" * (1024 * 1024)
            process.stdin.write(
                b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
            )
            process.stdin.flush()
            process.stdin.write(b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n')
            process.stdin.flush()
            assert select.select([process.stdout], [], [], 5)[0], (
                "No tools/list response"
            )
            tools_response = json.loads(process.stdout.readline())
            assert tools_response["id"] == 2 and tools_response["result"]["tools"]
        # Leave both pipes open, and the large response unread. communicate()
        # would drain stdout and mask a cancellation-blocked output worker.
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
        assert process.returncode == 143
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=3)

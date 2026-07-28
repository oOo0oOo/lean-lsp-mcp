"""Lean REPL for fast multi-attempt tactic execution using tactic mode."""

from __future__ import annotations

import asyncio
import json
import os
import platform
from dataclasses import dataclass, field
from typing import Any

from lean_lsp_mcp import config

if platform.system() != "Windows":
    import resource


class ReplError(Exception):
    pass


class ReplProcessError(ReplError):
    """The REPL process or protocol is unavailable, rather than Lean rejecting code."""


@dataclass
class SnippetResult:
    goals: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    proof_status: str | None = None
    error: str | None = None


@dataclass
class ReplRunResult:
    messages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    line_offset: int = 0


def repl_enabled() -> bool:
    return config.repl_enabled()


def find_repl_binary(project_dir: str | None = None) -> str | None:
    """Find REPL binary: env var > .lake/packages > PATH."""
    import shutil
    from pathlib import Path

    # 1. Explicit env var
    if path := config.repl_path():
        return path if Path(path).exists() or shutil.which(path) else None

    # 2. Auto-detect from .lake/packages (common location after `lake build`)
    if project_dir:
        candidates = [
            Path(project_dir)
            / ".lake"
            / "packages"
            / "repl"
            / ".lake"
            / "build"
            / "bin"
            / "repl",
            Path(project_dir)
            / ".lake"
            / "packages"
            / "REPL"
            / ".lake"
            / "build"
            / "bin"
            / "repl",
            Path(project_dir) / ".lake" / "build" / "bin" / "repl",
        ]
        for p in candidates:
            if p.exists():
                return str(p)

    # 3. Fall back to PATH
    if found := shutil.which("repl"):
        return found

    return None


def _split_imports_with_offset(code: str) -> tuple[str, str, int]:
    """Split code into imports, body, and the body's zero-based line offset."""
    lines = code.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith("import ")):
        i += 1

    # Deduplicate imports while preserving order
    imports = [ln.strip() for ln in lines[:i] if ln.startswith("import ")]
    header = "\n".join(dict.fromkeys(imports))
    return header, "\n".join(lines[i:]), i


def _split_imports(code: str) -> tuple[str, str]:
    """Split code into (header with imports, body)."""
    header, body, _ = _split_imports_with_offset(code)
    return header, body


def _memory_limit_preexec(mem_mb: int):
    """Return a subprocess memory-limit callback on supported platforms."""
    system = platform.system()
    if system == "Windows":
        return None
    limit = resource.RLIMIT_AS if system == "Linux" else resource.RLIMIT_RSS
    mem = mem_mb * 1024 * 1024

    def set_memory_limit() -> None:
        resource.setrlimit(limit, (mem, mem))

    return set_memory_limit


class Repl:
    """Lean REPL using tactic mode for fast multi-attempt."""

    def __init__(self, project_dir: str, repl_path: str | None = None):
        self.project_dir = project_dir
        self.repl_path = repl_path or find_repl_binary(project_dir) or "repl"
        self.timeout = config.repl_timeout()
        self.mem_mb = config.repl_mem_mb()
        self._proc: asyncio.subprocess.Process | None = None
        self._header: str | None = None
        self._header_env: int | None = None
        self._lock = asyncio.Lock()

    async def _start(self) -> None:
        """Start REPL subprocess."""
        kwargs: dict[str, Any] = {
            "cwd": self.project_dir,
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if platform.system() != "Windows":
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = _memory_limit_preexec(self.mem_mb)
        self._proc = await asyncio.create_subprocess_exec(
            "lake", "env", self.repl_path, **kwargs
        )

    async def _process_error(self, reason: str) -> ReplProcessError:
        """Describe a dead or unresponsive REPL without hiding its exit details."""
        proc = self._proc
        if proc is None:
            return ReplProcessError(reason)

        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=0.2)
        except asyncio.TimeoutError:
            pass

        stderr = ""
        if proc.stderr is not None and proc.returncode is not None:
            try:
                raw_stderr = await asyncio.wait_for(proc.stderr.read(), timeout=0.2)
                stderr = raw_stderr.decode(errors="replace").strip()
            except asyncio.TimeoutError:
                pass

        details = [reason]
        if proc.returncode is not None:
            details.append(f"exit code {proc.returncode}")
        if stderr:
            details.append(f"stderr: {stderr[-2000:]}")
        details.append(f"memory limit: {self.mem_mb} MiB ({config.REPL_MEM_MB_ENV})")
        return ReplProcessError("; ".join(details))

    async def _send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """Send command and return response."""
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            raise ReplProcessError("REPL not running")

        try:
            self._proc.stdin.write((json.dumps(cmd) + "\n\n").encode())
            await self._proc.stdin.drain()

            lines = []
            while True:
                line = await self._proc.stdout.readline()
                if not line or not line.strip():
                    break
                lines.append(line)

            if not lines:
                raise await self._process_error("No response from REPL")
            return json.loads(b"".join(lines))
        except ReplProcessError:
            raise
        except Exception as exc:
            raise await self._process_error(f"REPL command failed: {exc}") from exc

    async def _send_cmd(self, code: str, env: int | None = None) -> dict[str, Any]:
        """Send a command (code) to the REPL."""
        cmd: dict[str, Any] = {"cmd": code}
        if env is not None:
            cmd["env"] = env
        return await self._send(cmd)

    async def _send_tactic(self, tactic: str, proof_state: int) -> dict[str, Any]:
        """Send a tactic to run in a proof state."""
        return await self._send({"tactic": tactic, "proofState": proof_state})

    async def _ensure_header(self, header: str) -> int | None:
        """Ensure REPL is running with given header, return header env."""
        if self._header != header:
            await self.close()
            self._header = header
            self._header_env = None

        if not self._proc or self._proc.returncode is not None:
            await self._start()
            if header:
                resp = await self._send_cmd(header, env=None)
                if "error" in resp:
                    raise ReplError(f"Failed to load imports: {resp['error']}")
                self._header_env = resp.get("env")

        return self._header_env

    async def run_code(self, code: str) -> ReplRunResult:
        """Elaborate self-contained code against a cached import environment."""
        header, body, line_offset = _split_imports_with_offset(code)

        async with self._lock:
            try:
                header_env = await asyncio.wait_for(
                    self._ensure_header(header), timeout=self.timeout
                )
                if not body.strip():
                    return ReplRunResult(line_offset=line_offset)

                resp = await asyncio.wait_for(
                    self._send_cmd(body, env=header_env),
                    timeout=self.timeout,
                )
                return ReplRunResult(
                    messages=resp.get("messages", []),
                    error=resp.get("error"),
                    line_offset=line_offset,
                )
            except ReplError:
                await self.close()
                raise
            except asyncio.TimeoutError as exc:
                await self.close()
                raise ReplProcessError(
                    f"REPL command timed out after {self.timeout}s"
                ) from exc
            except Exception as exc:
                await self.close()
                raise ReplProcessError(str(exc)) from exc

    async def run_snippets(
        self, base_code: str, snippets: list[str]
    ) -> list[SnippetResult]:
        """Run multiple tactic snippets using tactic mode.

        1. Load header (imports) - cached across calls
        2. Send body + sorry to get proofState
        3. Run each tactic via tactic mode (very fast)
        """
        header, body = _split_imports(base_code)

        async with self._lock:
            try:
                # Load imports (cached)
                header_env = await asyncio.wait_for(
                    self._ensure_header(header), timeout=self.timeout
                )

                # Send body with sorry to get proof state
                if not body.strip():
                    return [SnippetResult(error="No proof body") for _ in snippets]

                # Match indentation of the last non-empty body line
                body_lines = body.splitlines()
                last_line = next((ln for ln in reversed(body_lines) if ln.strip()), "")
                indent = last_line[: len(last_line) - len(last_line.lstrip())]
                body_with_sorry = body.rstrip() + "\n" + indent + "sorry"
                resp = await asyncio.wait_for(
                    self._send_cmd(body_with_sorry, env=header_env),
                    timeout=self.timeout,
                )

                if "error" in resp:
                    return [SnippetResult(error=resp["error"]) for _ in snippets]

                # Get proof state from the sorry (use last — ours is always
                # appended at the end; earlier entries may be pre-existing).
                sorries = resp.get("sorries", [])
                if not sorries:
                    # No sorry = no proof goal, check messages for errors
                    msgs = resp.get("messages", [])
                    err = "; ".join(
                        m.get("data", "") for m in msgs if m.get("severity") == "error"
                    )
                    return [
                        SnippetResult(error=err or "No proof goal found")
                        for _ in snippets
                    ]

                proof_state = sorries[-1].get("proofState")
                if proof_state is None:
                    return [
                        SnippetResult(error="No proofState returned") for _ in snippets
                    ]

                # Run each tactic in tactic mode
                results = []
                for snippet in snippets:
                    try:
                        resp = await asyncio.wait_for(
                            self._send_tactic(snippet.strip(), proof_state),
                            timeout=self.timeout,
                        )

                        if "error" in resp:
                            # Lean error (tactic failed)
                            results.append(SnippetResult(error=resp["error"]))
                        else:
                            goals = resp.get("goals", [])
                            messages = resp.get("messages", [])
                            proof_status = resp.get("proofStatus")
                            results.append(
                                SnippetResult(
                                    goals=goals,
                                    messages=messages,
                                    proof_status=proof_status,
                                )
                            )
                    except ReplError:
                        raise
                    except asyncio.TimeoutError as exc:
                        raise ReplProcessError(
                            f"REPL tactic timed out after {self.timeout}s"
                        ) from exc
                    except Exception as e:
                        raise ReplProcessError(str(e)) from e

                return results

            except ReplError:
                await self.close()
                raise
            except asyncio.TimeoutError as exc:
                await self.close()
                raise ReplProcessError(
                    f"REPL command timed out after {self.timeout}s"
                ) from exc
            except Exception as e:
                await self.close()
                raise ReplProcessError(str(e)) from e

    async def close(self) -> None:
        if not self._proc:
            return
        proc, self._proc = self._proc, None
        self._header = None
        self._header_env = None
        try:
            if platform.system() != "Windows":
                os.killpg(os.getpgid(proc.pid), 9)
            else:
                proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            transport.close()

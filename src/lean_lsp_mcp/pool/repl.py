"""REPL subprocess wrapper."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    env: int | None = None
    sorries: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class ReplError(Exception):
    pass


class Repl:
    """Single REPL subprocess."""

    def __init__(self, header: str, repl_path: str, project_dir: str, mem_mb: int):
        self.header = header
        self.repl_path = repl_path
        self.project_dir = project_dir
        self.mem_mb = mem_mb
        self.proc: asyncio.subprocess.Process | None = None
        self.header_env: int | None = None

    async def start(self) -> None:
        kwargs: dict[str, Any] = {
            "cwd": self.project_dir,
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }

        system = platform.system()
        if system != "Windows":
            kwargs["start_new_session"] = True
            if system == "Linux":
                mem_bytes = self.mem_mb * 1024 * 1024

                def _set_mem_limit() -> None:
                    import resource

                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

                kwargs["preexec_fn"] = _set_mem_limit

        self.proc = await asyncio.create_subprocess_exec(
            "lake", "env", self.repl_path, **kwargs
        )
        logger.debug("Started REPL %x", id(self))

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def send(
        self, code: str, env: int | None = None, gc: bool = False
    ) -> CommandResult:
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            raise ReplError("REPL not started")

        cmd: dict[str, Any] = {"cmd": code}
        if env is not None:
            cmd["env"] = env
        if gc:
            cmd["gc"] = True

        payload = (json.dumps(cmd, ensure_ascii=False) + "\n\n").encode()

        try:
            self.proc.stdin.write(payload)
            await self.proc.stdin.drain()
        except BrokenPipeError as e:
            raise ReplError("Broken pipe") from e

        lines: list[bytes] = []
        while True:
            chunk = await self.proc.stdout.readline()
            if not chunk or not chunk.strip():
                break
            lines.append(chunk)

        try:
            resp = json.loads(b"".join(lines))
        except json.JSONDecodeError as e:
            raise ReplError("Invalid JSON response") from e

        return CommandResult(
            env=resp.get("env"),
            sorries=resp.get("sorries", []),
            messages=resp.get("messages", []),
            error=resp.get("error"),
        )

    async def close(self) -> None:
        if not self.proc:
            return
        proc = self.proc
        self.proc = None

        # Kill process
        try:
            if platform.system() != "Windows":
                os.killpg(os.getpgid(proc.pid), 9)
            else:
                proc.kill()
        except (ProcessLookupError, OSError):
            pass

        # Wait for exit and transport cleanup
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass

        # Close transport to prevent __del__ warning
        if hasattr(proc, "_transport") and proc._transport:
            proc._transport.close()

        logger.debug("Closed REPL %x", id(self))

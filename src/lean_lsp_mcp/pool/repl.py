"""REPL subprocess management for the pool.

Adapted from kimina-lean-server (MIT licensed).
Original: https://github.com/project-numina/kimina-lean-server
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import signal
import tempfile
from asyncio.subprocess import Process
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


def is_blank(s: str) -> bool:
    """Check if a string is blank (empty or whitespace only)."""
    return not s.strip()


class ReplError(Exception):
    """Error from REPL subprocess."""

    pass


class LeanError(Exception):
    """Error from Lean compiler."""

    pass


class Command(TypedDict, total=False):
    """Command sent to REPL subprocess."""

    cmd: str
    env: int
    gc: bool
    infotree: dict[str, Any]


@dataclass
class CommandResponse:
    """Response from REPL command execution."""

    env: int | None = None
    sorries: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    tactics: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class ReplResponse:
    """Full response including diagnostics."""

    id: str
    response: CommandResponse | None
    time: float
    diagnostics: dict[str, Any] | None = None
    error: str | None = None


class Repl:
    """Manages a single Lean REPL subprocess."""

    def __init__(
        self,
        uuid: UUID,
        created_at: datetime,
        header: str = "",
        *,
        max_repl_mem: int,
        max_repl_uses: int,
        repl_path: str,
        project_dir: str,
    ) -> None:
        self.uuid = uuid
        self.header = header
        self.use_count = 0
        self.created_at = created_at
        self.last_check_at = created_at

        # Stores the response received when running the import header.
        self.header_cmd_response: ReplResponse | None = None

        self.proc: Process | None = None
        self.error_file = tempfile.TemporaryFile("w+")
        self.max_memory_bytes = max_repl_mem * 1024 * 1024
        self.max_repl_uses = max_repl_uses
        self.repl_path = repl_path
        self.project_dir = project_dir

        self._loop: asyncio.AbstractEventLoop | None = None

        # REPL statistics (simplified - no psutil monitoring)
        self.cpu_per_exec: dict[int, float] = {}
        self.mem_per_exec: dict[int, int] = {}

    @classmethod
    async def create(
        cls,
        header: str,
        max_repl_uses: int,
        max_repl_mem: int,
        repl_path: str,
        project_dir: str,
    ) -> Repl:
        """Create a new REPL instance."""
        return cls(
            uuid=uuid4(),
            created_at=datetime.now(),
            header=header,
            max_repl_uses=max_repl_uses,
            max_repl_mem=max_repl_mem,
            repl_path=repl_path,
            project_dir=project_dir,
        )

    @property
    def exhausted(self) -> bool:
        """Check if REPL has exceeded its use limit."""
        if self.max_repl_uses < 0:
            return False
        if self.header and not is_blank(self.header):
            # Header does not count towards uses.
            return self.use_count >= self.max_repl_uses + 1
        return self.use_count >= self.max_repl_uses

    async def start(self) -> None:
        """Start the REPL subprocess."""
        self._loop = asyncio.get_running_loop()

        def _preexec() -> None:
            import resource

            # Memory limit (Linux only)
            if platform.system() != "Darwin":
                resource.setrlimit(
                    resource.RLIMIT_AS, (self.max_memory_bytes, self.max_memory_bytes)
                )

            os.setsid()

        self.proc = await asyncio.create_subprocess_exec(
            "lake",
            "env",
            self.repl_path,
            cwd=self.project_dir,
            env=os.environ,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_preexec,
        )

        logger.info(f"[{self.uuid.hex[:8]}] Started REPL subprocess")

    @property
    def is_running(self) -> bool:
        """Check if the subprocess is still running."""
        if not self.proc:
            return False
        return self.proc.returncode is None

    async def send_timeout(
        self,
        snippet_id: str,
        code: str,
        timeout: float,
        is_header: bool = False,
        env: int | None = None,
    ) -> ReplResponse:
        """Send a command with timeout handling."""
        try:
            cmd_response, elapsed_time = await asyncio.wait_for(
                self.send(snippet_id, code, is_header=is_header, env=env),
                timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            logger.error(
                "[%s] Lean REPL command timed out in %s seconds",
                self.uuid.hex[:8],
                timeout,
            )
            raise e
        except (LeanError, ReplError) as e:
            logger.exception("REPL error: %s", e)
            raise e

        return ReplResponse(
            id=snippet_id,
            response=cmd_response,
            time=elapsed_time,
            diagnostics={"repl_uuid": str(self.uuid)},
        )

    async def send(
        self,
        snippet_id: str,
        code: str,
        is_header: bool = False,
        env: int | None = None,
    ) -> tuple[CommandResponse, float]:
        """Send a command to the REPL subprocess.

        Args:
            snippet_id: Identifier for logging
            code: Lean code to execute
            is_header: If True, this is the initial import header
            env: Environment ID to fork from (for backtracking)
        """
        logger.debug("[%s] Running snippet %s", self.uuid.hex[:8], snippet_id)

        if not self.proc or self.proc.returncode is not None:
            logger.error("REPL process not started or shut down")
            raise ReplError("REPL process not started or shut down")

        loop = self._loop or asyncio.get_running_loop()

        if self.proc.stdin is None:
            raise ReplError("stdin pipe not initialized")
        if self.proc.stdout is None:
            raise ReplError("stdout pipe not initialized")

        input_cmd: Command = {"cmd": code}

        # If env specified, use it for backtracking (fork from that state)
        if env is not None:
            input_cmd["env"] = env
        # GC after first use (not for header) when no specific env requested
        elif self.use_count != 0 and not is_header:
            input_cmd["env"] = 0
            input_cmd["gc"] = True

        payload = (json.dumps(input_cmd, ensure_ascii=False) + "\n\n").encode("utf-8")

        start = loop.time()
        logger.debug("Sending payload to REPL")

        try:
            self.proc.stdin.write(payload)
            await self.proc.stdin.drain()
        except BrokenPipeError:
            logger.error("Broken pipe while writing to REPL stdin")
            raise LeanError("Lean process broken pipe")
        except Exception as e:
            logger.error("Failed to write to REPL stdin: %s", e)
            raise LeanError("Failed to write to REPL stdin")

        logger.debug("Reading response from REPL stdout")
        raw = await self._read_response()
        elapsed = loop.time() - start

        logger.debug("Raw response from REPL: %r", raw[:200] if raw else b"")
        try:
            resp: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("JSON decode error: %r", raw[:200] if raw else b"")
            raise ReplError("JSON decode error")

        self.error_file.seek(0)
        err = self.error_file.read().strip()
        self.error_file.seek(0)
        self.error_file.truncate(0)
        if err:
            logger.error("Stderr: %s", err)
            raise LeanError(err)

        elapsed_time = round(elapsed, 6)

        self.use_count += 1

        # Parse response into CommandResponse
        cmd_response = CommandResponse(
            env=resp.get("env"),
            sorries=resp.get("sorries", []),
            messages=resp.get("messages", []),
            tactics=resp.get("tactics", []),
            error=resp.get("error"),
        )

        return cmd_response, elapsed_time

    async def _read_response(self) -> bytes:
        """Read response from REPL stdout."""
        if not self.proc or self.proc.stdout is None:
            logger.error("REPL process not started or stdout pipe not initialized")
            raise ReplError("REPL process not started or stdout pipe not initialized")

        lines: list[bytes] = []
        try:
            while True:
                chunk = await self.proc.stdout.readline()
                # EOF or blank line as terminator
                if not chunk or not chunk.strip():
                    break
                lines.append(chunk)
        except Exception as e:
            logger.error("Failed to read from REPL stdout: %s", e)
            raise LeanError("Failed to read from REPL stdout")
        return b"".join(lines)

    async def close(self) -> None:
        """Close the REPL subprocess."""
        if self.proc:
            self.last_check_at = datetime.now()
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # Already dead
            await self.proc.wait()


async def close_verbose(repl: Repl) -> None:
    """Close a REPL with logging."""
    uuid = repl.uuid
    logger.info(f"Closing REPL {uuid.hex[:8]}")
    await repl.close()
    logger.info(f"Closed REPL {uuid.hex[:8]}")

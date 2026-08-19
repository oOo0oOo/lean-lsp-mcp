"""Build coordination and streamed Lake command execution."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from leanclient.aio import AsyncLeanLSPClient
from mcp.server.mcpserver.utilities.logging import get_logger

from lean_lsp_mcp import config
from lean_lsp_mcp.client_utils import (
    attach_shared_client,
    detach_shared_client,
    set_build_in_progress,
)
from lean_lsp_mcp.models import BuildResult
from lean_lsp_mcp.tool_utils import safe_report_progress

logger = get_logger(__name__)


class BuildCoordinator:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self._lock = asyncio.Lock()
        self._current_task: asyncio.Task[BuildResult] | None = None

    async def run(
        self, build_factory: Callable[[], Coroutine[Any, Any, BuildResult]]
    ) -> BuildResult:
        if self.mode == "allow":
            return await build_factory()

        async with self._lock:
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
            self._current_task = asyncio.create_task(build_factory())
            task = self._current_task

        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
            if self.mode == "cancel":
                return BuildResult(
                    success=False,
                    output="",
                    errors=["Build superseded by newer request."],
                )
            while True:
                latest = self._current_task
                try:
                    return await latest
                except asyncio.CancelledError:
                    if self._current_task is latest:
                        raise


@dataclass
class LakeBuildRunner:
    """Run Lake commands while collecting filtered output and progress."""

    ctx: Any
    log_lines: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    active_process: asyncio.subprocess.Process | None = None

    async def _handle_line(self, line: str) -> None:
        line = line.rstrip()
        if line.startswith("trace:") or "LEAN_PATH=" in line:
            return

        self.log_lines.append(line)
        if "error" in line.lower():
            self.errors.append(line)

        if match := re.search(
            r"\[(\d+)/(\d+)\]\s*(.+?)(?:\s+\(\d+\.?\d*[ms]+\))?$", line
        ):
            await safe_report_progress(
                self.ctx,
                progress=int(match.group(1)),
                total=int(match.group(2)),
                message=match.group(3) or "Building",
            )

    async def run(self, *args: str, cwd: Path) -> asyncio.subprocess.Process:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self.active_process = process
        assert process.stdout is not None

        remainder = ""
        while chunk := await process.stdout.read(64 * 1024):
            parts = (remainder + chunk.decode("utf-8", errors="replace")).split("\n")
            remainder = parts.pop()
            for line in parts:
                await self._handle_line(line)
        if remainder:
            await self._handle_line(remainder)

        await process.wait()
        return process

    def output(self, line_count: int) -> str:
        return "\n".join(self.log_lines[-line_count:]) if line_count else ""

    async def cancel(self) -> None:
        process = self.active_process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()


async def _stop_project_clients(ctx: Any, project_path: Path) -> None:
    session_client = ctx.request_context.lifespan_context.client
    ctx.request_context.lifespan_context.client = None
    shared_client = await detach_shared_client(project_path)

    clients = []
    for client in (session_client, shared_client):
        if client is not None and client not in clients:
            clients.append(client)
    for client in clients:
        try:
            await client.close()
        except Exception:
            logger.exception("Lean client close failed during lsp_build restart")


async def run_build(
    ctx: Any,
    project_path: Path,
    clean: bool,
    fetch_cache: bool,
    output_lines: int,
) -> BuildResult:
    runner = LakeBuildRunner(ctx)
    build_flag_set = False

    try:
        set_build_in_progress(project_path, True)
        build_flag_set = True
        await _stop_project_clients(ctx, project_path)

        if clean:
            await safe_report_progress(
                ctx, progress=1, total=16, message="Running `lake clean`"
            )
            await runner.run("lake", "clean", cwd=project_path)

        if fetch_cache:
            await safe_report_progress(
                ctx, progress=2, total=16, message="Running `lake exe cache get`"
            )
            await runner.run("lake", "exe", "cache", "get", cwd=project_path)

        process = await runner.run("lake", "build", cwd=project_path)
        if process.returncode != 0:
            return BuildResult(
                success=False,
                output=runner.output(output_lines),
                errors=runner.errors
                or [f"Build failed with return code {process.returncode}"],
            )

        client = AsyncLeanLSPClient(
            str(project_path),
            max_workers=config.max_open_files(),
        )
        await client.start()
        logger.info("Built project and re-started LSP client")
        attach_shared_client(project_path, client)
        ctx.request_context.lifespan_context.client = client
        return BuildResult(
            success=True,
            output=runner.output(output_lines),
            errors=[],
        )
    except asyncio.CancelledError:
        await runner.cancel()
        raise
    except Exception as exc:
        return BuildResult(
            success=False,
            output=runner.output(output_lines),
            errors=[str(exc)],
        )
    finally:
        if build_flag_set:
            set_build_in_progress(project_path, False)

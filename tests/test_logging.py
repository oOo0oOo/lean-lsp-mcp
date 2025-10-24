from __future__ import annotations

import sys
import tempfile
from typing import Awaitable, Callable

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from tests.conftest import _server_environment


async def _collect_logs(
    repo_root,
    env_overrides: dict[str, str],
    interaction: Callable[[ClientSession], Awaitable[None]],
) -> str:
    env = _server_environment(repo_root)
    env.update(env_overrides)

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errlog:
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "lean_lsp_mcp", "--transport", "stdio"],
            env=env,
            cwd=str(repo_root),
        )

        async with stdio_client(server, errlog=errlog) as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            async with session:
                await session.initialize()
                await interaction(session)

        errlog.seek(0)
        return errlog.read()


@pytest.mark.asyncio
async def test_no_stderr_when_log_level_none(repo_root, test_project_path) -> None:
    async def interaction(session: ClientSession) -> None:
        await session.list_tools()

    logs = await _collect_logs(
        repo_root,
        {
            "LEAN_LOG_LEVEL": "NONE",
            "LEAN_PROJECT_PATH": str(test_project_path),
        },
        interaction,
    )

    assert not logs.strip()


@pytest.mark.asyncio
async def test_info_level_emits_server_logs(repo_root, test_project_path) -> None:
    async def interaction(session: ClientSession) -> None:
        await session.list_tools()

    logs = await _collect_logs(
        repo_root,
        {
            "LEAN_LOG_LEVEL": "INFO",
            "LEAN_PROJECT_PATH": str(test_project_path),
        },
        interaction,
    )

    normalized = " ".join(logs.split())
    assert "Closing Lean LSP client" in normalized


@pytest.mark.asyncio
async def test_error_level_suppresses_info_logs(repo_root, test_project_path) -> None:
    async def interaction(session: ClientSession) -> None:
        await session.list_tools()

    logs = await _collect_logs(
        repo_root,
        {
            "LEAN_LOG_LEVEL": "ERROR",
            "LEAN_PROJECT_PATH": str(test_project_path),
        },
        interaction,
    )

    normalized = " ".join(logs.split())
    assert "Closing Lean LSP client" not in normalized

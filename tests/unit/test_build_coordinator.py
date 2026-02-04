from __future__ import annotations

import asyncio

import pytest

from lean_lsp_mcp.models import BuildResult
from lean_lsp_mcp.server import BuildCoordinator


@pytest.mark.asyncio
async def test_build_coordinator_cancel_mode() -> None:
    coordinator = BuildCoordinator("cancel")
    started = asyncio.Event()

    async def build_one() -> BuildResult:
        started.set()
        await asyncio.Event().wait()
        return BuildResult(success=True, output="one", errors=[])

    async def build_two() -> BuildResult:
        return BuildResult(success=True, output="two", errors=[])

    task_one = asyncio.create_task(coordinator.run(build_one))
    await started.wait()
    task_two = asyncio.create_task(coordinator.run(build_two))

    result_two = await task_two
    result_one = await task_one

    assert result_two.output == "two"
    assert result_one.success is False
    assert result_one.errors and "superseded" in result_one.errors[0].lower()


@pytest.mark.asyncio
async def test_build_coordinator_share_mode() -> None:
    coordinator = BuildCoordinator("share")
    started = asyncio.Event()

    async def build_one() -> BuildResult:
        started.set()
        await asyncio.Event().wait()
        return BuildResult(success=True, output="one", errors=[])

    async def build_two() -> BuildResult:
        return BuildResult(success=True, output="two", errors=[])

    task_one = asyncio.create_task(coordinator.run(build_one))
    await started.wait()
    task_two = asyncio.create_task(coordinator.run(build_two))

    result_one = await task_one
    result_two = await task_two

    assert result_one.output == "two"
    assert result_two.output == "two"

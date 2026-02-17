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


@pytest.mark.asyncio
async def test_build_coordinator_cancel_mode_many_callers_deterministic() -> None:
    coordinator = BuildCoordinator("cancel")
    caller_count = 5
    tasks: list[asyncio.Task[BuildResult]] = []

    for i in range(caller_count):
        started = asyncio.Event()

        if i < caller_count - 1:

            async def build(i: int = i, started: asyncio.Event = started) -> BuildResult:
                started.set()
                await asyncio.Event().wait()
                return BuildResult(success=True, output=f"build-{i}", errors=[])

        else:

            async def build(i: int = i, started: asyncio.Event = started) -> BuildResult:
                started.set()
                await asyncio.sleep(0)
                return BuildResult(success=True, output=f"build-{i}", errors=[])

        tasks.append(asyncio.create_task(coordinator.run(build)))
        await started.wait()

    results = await asyncio.gather(*tasks)
    final_output = f"build-{caller_count - 1}"
    superseded = [result for result in results if not result.success]
    succeeded = [result for result in results if result.success]

    assert len(superseded) == caller_count - 1
    assert len(succeeded) == 1
    assert succeeded[0].output == final_output
    assert all(result.errors and "superseded" in result.errors[0].lower() for result in superseded)
    assert coordinator._task_waiters == {}
    assert coordinator._superseded_waiters == {}


@pytest.mark.asyncio
async def test_build_coordinator_share_mode_many_callers_get_latest_result() -> None:
    coordinator = BuildCoordinator("share")
    caller_count = 5
    tasks: list[asyncio.Task[BuildResult]] = []
    final_output = f"build-{caller_count - 1}"

    for i in range(caller_count):
        started = asyncio.Event()

        if i < caller_count - 1:

            async def build(i: int = i, started: asyncio.Event = started) -> BuildResult:
                started.set()
                await asyncio.Event().wait()
                return BuildResult(success=True, output=f"build-{i}", errors=[])

        else:

            async def build(i: int = i, started: asyncio.Event = started) -> BuildResult:
                started.set()
                await asyncio.sleep(0)
                return BuildResult(success=True, output=f"build-{i}", errors=[])

        tasks.append(asyncio.create_task(coordinator.run(build)))
        await started.wait()

    results = await asyncio.gather(*tasks)
    assert all(result.success for result in results)
    assert all(result.output == final_output for result in results)
    assert coordinator._task_waiters == {}
    assert coordinator._superseded_waiters == {}

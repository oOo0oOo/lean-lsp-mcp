from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from lean_lsp_mcp import client_utils
from lean_lsp_mcp.client_utils import (
    attach_shared_client,
    bind_lean_project_path,
    close_shared_client,
    resolve_file_path,
    set_build_in_progress,
    setup_client_for_file,
    startup_client,
    valid_lean_project_path,
)


class _FakeTransport:
    def __init__(self) -> None:
        self.kill_calls = 0

    def _kill_group(self) -> None:
        self.kill_calls += 1


class _MockAioClient:
    def __init__(self, project_path: Path) -> None:
        self.project_path = str(project_path)
        self.closed = False
        self._alive = True
        self._transport = _FakeTransport()

    @property
    def alive(self) -> bool:
        return self._alive and not self.closed

    async def close(self) -> None:
        self.closed = True


class _FailingCloseClient(_MockAioClient):
    def __init__(self, project_path: Path) -> None:
        super().__init__(project_path)
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        raise PermissionError("operation not permitted")


class _LifespanContext:
    def __init__(
        self,
        lean_project_path: Path | None,
        client: _MockAioClient | None,
        *,
        active_transport: str = "stdio",
        project_switching_allowed: bool = True,
    ) -> None:
        self.lean_project_path = lean_project_path
        self.client = client
        self.active_transport = active_transport
        self.project_switching_allowed = project_switching_allowed
        self.file_content_hashes: dict[str, int] = {}
        self.project_cache: dict[str, Path | str] = {}


class _RequestContext:
    def __init__(self, lifespan_context: _LifespanContext) -> None:
        self.lifespan_context = lifespan_context


class _Context:
    def __init__(self, lifespan_context: _LifespanContext) -> None:
        self.request_context = _RequestContext(lifespan_context)


def _make_project(root: Path) -> Path:
    root.mkdir()
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    (root / "lakefile.toml").write_text('name = "test"\n')
    return root


def _make_dependency(project: Path, dep_root: Path) -> Path:
    dep_file = dep_root / "Mathlib" / "Foo.lean"
    dep_file.parent.mkdir(parents=True)
    dep_file.write_text("theorem dep : True := by trivial\n")

    dep_link = project / ".lake" / "packages" / "mathlib"
    dep_link.parent.mkdir(parents=True)
    dep_link.symlink_to(dep_root, target_is_directory=True)
    return dep_file


@pytest.fixture(autouse=True)
def _reset_shared_clients() -> None:
    client_utils._shared_clients.clear()
    client_utils._shared_pools.clear()
    client_utils._builds_in_progress.clear()
    yield
    client_utils._shared_clients.clear()
    client_utils._shared_pools.clear()
    client_utils._builds_in_progress.clear()


@pytest.fixture
def patched_clients(monkeypatch: pytest.MonkeyPatch) -> list[_MockAioClient]:
    created: list[_MockAioClient] = []

    async def _constructor(project_path: Path) -> _MockAioClient:
        client = _MockAioClient(project_path)
        created.append(client)
        return client

    monkeypatch.setattr(client_utils, "_start_client", _constructor)
    return created


@pytest.mark.asyncio
async def test_startup_client_reuses_existing(
    tmp_path: Path, patched_clients: list[_MockAioClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    ctx = _Context(_LifespanContext(project, None))

    await startup_client(ctx)
    first = ctx.request_context.lifespan_context.client
    assert isinstance(first, _MockAioClient)
    assert not first.closed

    await startup_client(ctx)
    assert ctx.request_context.lifespan_context.client is first
    assert len(patched_clients) == 1


@pytest.mark.asyncio
async def test_startup_client_keeps_project_specific_clients(
    tmp_path: Path, patched_clients: list[_MockAioClient]
) -> None:
    project1 = _make_project(tmp_path / "proj1")
    project2 = _make_project(tmp_path / "proj2")
    ctx = _Context(_LifespanContext(project1, None))

    await startup_client(ctx)
    first = ctx.request_context.lifespan_context.client

    ctx.request_context.lifespan_context.lean_project_path = project2
    await startup_client(ctx)
    second = ctx.request_context.lifespan_context.client

    ctx.request_context.lifespan_context.lean_project_path = project1
    await startup_client(ctx)

    assert first is not None
    assert second is not None
    assert first is not second
    assert ctx.request_context.lifespan_context.client is first
    assert not first.closed
    assert not second.closed
    assert len(patched_clients) == 2


@pytest.mark.asyncio
async def test_startup_client_recreates_dead_shared_client(
    tmp_path: Path, patched_clients: list[_MockAioClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    ctx = _Context(_LifespanContext(project, None))

    await startup_client(ctx)
    first = ctx.request_context.lifespan_context.client
    first._alive = False

    await startup_client(ctx)

    second = ctx.request_context.lifespan_context.client
    assert second is not first
    assert first.closed
    assert len(patched_clients) == 2


@pytest.mark.asyncio
async def test_startup_client_recreates_dead_client_even_if_close_fails(
    tmp_path: Path, patched_clients: list[_MockAioClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    old_client = _FailingCloseClient(project)
    old_client._alive = False

    attach_shared_client(project, old_client)

    ctx = _Context(_LifespanContext(project, old_client))
    await startup_client(ctx)

    new_client = ctx.request_context.lifespan_context.client
    assert isinstance(new_client, _MockAioClient)
    assert new_client is not old_client
    assert Path(new_client.project_path) == project.resolve()
    assert old_client.close_calls == 1
    assert len(patched_clients) == 1


def test_valid_lean_project_path_requires_lakefile(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    assert not valid_lean_project_path(project)
    assert not valid_lean_project_path(project / "missing")

    (project / "lakefile.lean").write_text("import Lake\n")
    assert valid_lean_project_path(project)


@pytest.mark.asyncio
async def test_setup_client_for_file_discovers_project(
    tmp_path: Path, patched_clients: list[_MockAioClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    lean_file = project / "src" / "Example.lean"
    lean_file.parent.mkdir(parents=True)
    lean_file.write_text("example")

    ctx = _Context(_LifespanContext(None, None))

    rel_path = await setup_client_for_file(ctx, str(lean_file))
    assert rel_path == "src/Example.lean"
    assert (
        Path(ctx.request_context.lifespan_context.client.project_path)
        == project.resolve()
    )
    assert len(patched_clients) == 1


@pytest.mark.asyncio
async def test_setup_client_for_file_reuses_client_for_same_project(
    tmp_path: Path, patched_clients: list[_MockAioClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    file1 = project / "File1.lean"
    file1.write_text("theorem a : True := by trivial")
    file2 = project / "src" / "File2.lean"
    file2.parent.mkdir(parents=True)
    file2.write_text("theorem b : True := by trivial")

    ctx = _Context(_LifespanContext(None, None))

    rel_path1 = await setup_client_for_file(ctx, str(file1))
    assert rel_path1 == "File1.lean"
    first_client = ctx.request_context.lifespan_context.client
    assert len(patched_clients) == 1

    rel_path2 = await setup_client_for_file(ctx, str(file2))
    assert rel_path2 == "src/File2.lean"
    assert ctx.request_context.lifespan_context.client is first_client
    assert not first_client.closed
    assert len(patched_clients) == 1


@pytest.mark.asyncio
async def test_setup_client_for_file_switches_projects_without_closing_cache(
    tmp_path: Path, patched_clients: list[_MockAioClient]
) -> None:
    project1 = _make_project(tmp_path / "proj1")
    file1 = project1 / "File1.lean"
    file1.write_text("theorem a : True := by trivial")

    project2 = _make_project(tmp_path / "proj2")
    file2 = project2 / "File2.lean"
    file2.write_text("theorem b : True := by trivial")

    ctx = _Context(_LifespanContext(None, None))

    assert await setup_client_for_file(ctx, str(file1)) == "File1.lean"
    first_client = ctx.request_context.lifespan_context.client

    assert await setup_client_for_file(ctx, str(file2)) == "File2.lean"
    second_client = ctx.request_context.lifespan_context.client

    assert first_client is not second_client
    assert not first_client.closed
    assert not second_client.closed
    assert len(patched_clients) == 2
    assert ctx.request_context.lifespan_context.lean_project_path == project2


@pytest.mark.asyncio
async def test_setup_client_for_dependency_file_uses_parent_project(
    tmp_path: Path, patched_clients: list[_MockAioClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    try:
        dep_file = _make_dependency(project, tmp_path / "deps" / "mathlib")
    except OSError as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"symlinks unavailable: {exc}")

    ctx = _Context(_LifespanContext(project, None))

    rel_path = await setup_client_for_file(ctx, str(dep_file))

    assert rel_path == os.path.relpath(dep_file, project)
    assert ctx.request_context.lifespan_context.lean_project_path == project
    assert (
        Path(ctx.request_context.lifespan_context.client.project_path)
        == project.resolve()
    )
    assert len(patched_clients) == 1


def test_bind_lean_project_path_rejects_switch_on_remote_transport(
    tmp_path: Path,
) -> None:
    project1 = _make_project(tmp_path / "proj1")
    project2 = _make_project(tmp_path / "proj2")
    ctx = _Context(
        _LifespanContext(
            project1,
            None,
            active_transport="streamable-http",
            project_switching_allowed=False,
        )
    )

    with pytest.raises(ValueError, match="Project switching is disabled"):
        bind_lean_project_path(ctx, project2)


@pytest.mark.asyncio
async def test_startup_client_serializes_concurrent_calls(
    tmp_path: Path, patched_clients: list[_MockAioClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    ctx = _Context(_LifespanContext(project, None))

    await asyncio.gather(*(startup_client(ctx) for _ in range(10)))

    assert len(patched_clients) == 1
    assert ctx.request_context.lifespan_context.client is patched_clients[0]


@pytest.mark.asyncio
async def test_build_in_progress_blocks_only_same_project(
    tmp_path: Path, patched_clients: list[_MockAioClient]
) -> None:
    project1 = _make_project(tmp_path / "proj1")
    project2 = _make_project(tmp_path / "proj2")

    set_build_in_progress(project1, True)

    try:
        with pytest.raises(ValueError, match="build is in progress"):
            await startup_client(_Context(_LifespanContext(project1, None)))

        other_ctx = _Context(_LifespanContext(project2, None))
        await startup_client(other_ctx)

        assert other_ctx.request_context.lifespan_context.client is patched_clients[0]
        assert len(patched_clients) == 1
    finally:
        set_build_in_progress(project1, False)


def test_close_shared_client_kills_process_group(tmp_path: Path) -> None:
    """close_shared_client() is the post-event-loop exit path: it must
    terminate synchronously via the transport's process-group kill."""
    project = _make_project(tmp_path / "proj")
    client = _MockAioClient(project)
    attach_shared_client(project, client)

    close_shared_client()

    assert client._transport.kill_calls == 1
    assert client_utils._shared_clients == {}


def test_close_shared_client_suppresses_error(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "proj")
    client = _MockAioClient(project)

    def _boom() -> None:
        raise PermissionError("operation not permitted")

    client._transport._kill_group = _boom
    attach_shared_client(project, client)

    close_shared_client()  # should not raise
    assert client_utils._shared_clients == {}


def test_resolve_file_path_uses_project_root_for_relative(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "proj")
    target = project / "src" / "Example.lean"
    target.parent.mkdir(parents=True)
    target.write_text("theorem t : True := by trivial")

    ctx = _Context(_LifespanContext(project, None))
    resolved = resolve_file_path(ctx, "src/Example.lean")
    assert resolved == target.resolve()


@pytest.mark.asyncio
async def test_open_synced_reads_utf8_without_reload_from_disk(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "proj")
    source = "-- Unicode: \u2212 \u2080 \u03b1\ntheorem clean : True := trivial\n"
    (project / "Source.lean").write_text(source, encoding="utf-8")

    class _Client:
        project_path = str(project)

        def __init__(self) -> None:
            self.open_calls: list[tuple[str, str, bool]] = []

        async def open(self, path: str, text: str, wait: bool = False):
            self.open_calls.append((path, text, wait))
            return object()

        async def reload_from_disk(self, *_args, **_kwargs):
            raise AssertionError("open_synced must not use the locale-dependent reload")

    client = _Client()
    ctx = _Context(_LifespanContext(project, client))

    for _ in range(3):
        await client_utils.open_synced(ctx, "Source.lean", wait=True)

    assert client.open_calls == [("Source.lean", source, True)] * 3

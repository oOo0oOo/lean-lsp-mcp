from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lean_lsp_mcp.client_utils import (
    bind_lean_project_path,
    resolve_file_path,
    setup_client_for_file,
    startup_client,
    valid_lean_project_path,
)


class _MockLeanClient:
    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FailingCloseClient(_MockLeanClient):
    def __init__(self, project_path: Path) -> None:
        super().__init__(project_path)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise PermissionError("operation not permitted")


class _LifespanContext:
    def __init__(
        self,
        lean_project_path: Path | None,
        client: _MockLeanClient | None,
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


@pytest.fixture
def patched_clients(monkeypatch: pytest.MonkeyPatch) -> list[_MockLeanClient]:
    created: list[_MockLeanClient] = []

    def _constructor(
        project_path: Path, initial_build: bool, prevent_cache_get: bool = False
    ) -> _MockLeanClient:  # pragma: no cover - signature verified indirectly
        _ = initial_build, prevent_cache_get
        client = _MockLeanClient(project_path)
        created.append(client)
        return client

    monkeypatch.setattr("lean_lsp_mcp.client_utils.LeanLSPClient", _constructor)
    return created


def test_startup_client_reuses_existing(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    ctx = _Context(_LifespanContext(project, None))

    startup_client(ctx)
    first = ctx.request_context.lifespan_context.client
    assert isinstance(first, _MockLeanClient)
    assert not first.closed

    startup_client(ctx)
    assert not first.closed

    new_project = _make_project(tmp_path / "proj2")
    ctx.request_context.lifespan_context.lean_project_path = new_project

    startup_client(ctx)
    assert first.closed
    assert ctx.request_context.lifespan_context.client.project_path == new_project
    assert len(patched_clients) == 2


def test_startup_client_switches_project_even_if_close_fails(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    old_project = _make_project(tmp_path / "proj1")
    new_project = _make_project(tmp_path / "proj2")

    old_client = _FailingCloseClient(old_project)
    ctx = _Context(_LifespanContext(new_project, old_client))

    startup_client(ctx)

    new_client = ctx.request_context.lifespan_context.client
    assert isinstance(new_client, _MockLeanClient)
    assert new_client is not old_client
    assert new_client.project_path == new_project
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


def test_setup_client_for_file_discovers_project(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    lean_file = project / "src" / "Example.lean"
    lean_file.parent.mkdir(parents=True)
    lean_file.write_text("example")

    ctx = _Context(_LifespanContext(None, None))

    rel_path = setup_client_for_file(ctx, str(lean_file))
    assert rel_path == "src/Example.lean"
    assert ctx.request_context.lifespan_context.client.project_path == project
    assert len(patched_clients) == 1


def test_setup_client_for_file_reuses_client_for_same_project(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    file1 = project / "File1.lean"
    file1.write_text("theorem a : True := by trivial")
    file2 = project / "src" / "File2.lean"
    file2.parent.mkdir(parents=True)
    file2.write_text("theorem b : True := by trivial")

    ctx = _Context(_LifespanContext(None, None))

    rel_path1 = setup_client_for_file(ctx, str(file1))
    assert rel_path1 == "File1.lean"
    first_client = ctx.request_context.lifespan_context.client
    assert len(patched_clients) == 1

    rel_path2 = setup_client_for_file(ctx, str(file2))
    assert rel_path2 == "src/File2.lean"
    assert ctx.request_context.lifespan_context.client is first_client
    assert not first_client.closed
    assert len(patched_clients) == 1


def test_setup_client_for_file_switches_projects(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project1 = _make_project(tmp_path / "proj1")
    file1 = project1 / "File1.lean"
    file1.write_text("theorem a : True := by trivial")

    project2 = _make_project(tmp_path / "proj2")
    file2 = project2 / "File2.lean"
    file2.write_text("theorem b : True := by trivial")

    ctx = _Context(_LifespanContext(None, None))

    rel_path1 = setup_client_for_file(ctx, str(file1))
    assert rel_path1 == "File1.lean"
    first_client = ctx.request_context.lifespan_context.client
    assert len(patched_clients) == 1

    rel_path2 = setup_client_for_file(ctx, str(file2))
    assert rel_path2 == "File2.lean"
    second_client = ctx.request_context.lifespan_context.client

    assert first_client.closed
    assert second_client is not first_client
    assert len(patched_clients) == 2
    assert ctx.request_context.lifespan_context.lean_project_path == project2


def test_setup_client_for_dependency_file_uses_parent_project(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    dep_file = _make_dependency(project, tmp_path / "deps" / "mathlib")

    ctx = _Context(_LifespanContext(project, None))

    rel_path = setup_client_for_file(ctx, str(dep_file))

    assert rel_path == os.path.relpath(dep_file, project)
    assert ctx.request_context.lifespan_context.lean_project_path == project
    assert ctx.request_context.lifespan_context.client.project_path == project
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


def test_startup_client_serializes_concurrent_calls(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = _make_project(tmp_path / "proj")
    ctx = _Context(_LifespanContext(project, None))

    def _invoke_startup() -> None:
        startup_client(ctx)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_invoke_startup) for _ in range(10)]
        for future in futures:
            assert future.result() is None

    assert len(patched_clients) == 1
    assert ctx.request_context.lifespan_context.client is patched_clients[0]


def test_resolve_file_path_uses_project_root_for_relative(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "proj")
    target = project / "src" / "Example.lean"
    target.parent.mkdir(parents=True)
    target.write_text("theorem t : True := by trivial")

    ctx = _Context(_LifespanContext(project, None))
    resolved = resolve_file_path(ctx, "src/Example.lean")
    assert resolved == target.resolve()

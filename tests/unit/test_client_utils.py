from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lean_lsp_mcp.client_utils import (
    setup_client_for_file,
    startup_client,
    valid_lean_project_path,
)


class _MockLeanClient:
    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path
        self.closed = False
        self.opened_files: dict[str, str] = {}
        self.open_calls: list[tuple[str, bool]] = []
        self.update_calls: list[tuple[str, str]] = []

    def close(self) -> None:
        self.closed = True

    def open_file(self, path: str, force_reopen: bool = False) -> None:
        self.open_calls.append((path, force_reopen))
        abs_path = self.project_path / path
        self.opened_files[path] = abs_path.read_text(encoding="utf-8")

    def get_file_content(self, path: str) -> str:
        if path not in self.opened_files:
            raise FileNotFoundError(path)
        return self.opened_files[path]

    def update_file_content(self, path: str, content: str) -> None:
        self.update_calls.append((path, content))
        self.opened_files[path] = content


class _FailingCloseClient(_MockLeanClient):
    def __init__(self, project_path: Path) -> None:
        super().__init__(project_path)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise PermissionError("operation not permitted")


class _LifespanContext:
    def __init__(
        self, lean_project_path: Path | None, client: _MockLeanClient | None
    ) -> None:
        self.lean_project_path = lean_project_path
        self.client = client
        self.file_content_hashes: dict[str, int] = {}


class _RequestContext:
    def __init__(self, lifespan_context: _LifespanContext) -> None:
        self.lifespan_context = lifespan_context


class _Context:
    def __init__(self, lifespan_context: _LifespanContext) -> None:
        self.request_context = _RequestContext(lifespan_context)


@pytest.fixture
def patched_clients(monkeypatch: pytest.MonkeyPatch) -> list[_MockLeanClient]:
    created: list[_MockLeanClient] = []

    def _constructor(
        project_path: Path, initial_build: bool, prevent_cache_get: bool = False
    ) -> _MockLeanClient:  # pragma: no cover - signature verified indirectly
        client = _MockLeanClient(project_path)
        created.append(client)
        return client

    monkeypatch.setattr("lean_lsp_mcp.client_utils.LeanLSPClient", _constructor)
    return created


def test_startup_client_reuses_existing(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    ctx = _Context(_LifespanContext(project, None))

    startup_client(ctx)
    first = ctx.request_context.lifespan_context.client
    assert isinstance(first, _MockLeanClient)
    assert not first.closed

    # second startup with same project path should reuse existing client
    startup_client(ctx)
    assert not first.closed

    # change project path triggers close and replacement
    new_project = tmp_path / "proj2"
    new_project.mkdir()
    (new_project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    ctx.request_context.lifespan_context.lean_project_path = new_project

    startup_client(ctx)
    assert first.closed
    assert ctx.request_context.lifespan_context.client.project_path == new_project
    assert len(patched_clients) == 2


def test_startup_client_switches_project_even_if_close_fails(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    old_project = tmp_path / "proj1"
    old_project.mkdir()
    (old_project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    new_project = tmp_path / "proj2"
    new_project.mkdir()
    (new_project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    old_client = _FailingCloseClient(old_project)
    ctx = _Context(_LifespanContext(new_project, old_client))

    startup_client(ctx)

    new_client = ctx.request_context.lifespan_context.client
    assert isinstance(new_client, _MockLeanClient)
    assert new_client is not old_client
    assert new_client.project_path == new_project
    assert old_client.close_calls == 1
    assert len(patched_clients) == 1


def test_valid_lean_project_path(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0")

    assert valid_lean_project_path(project)
    assert not valid_lean_project_path(project / "missing")


def test_setup_client_for_file_discovers_project(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

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
    """Verify that multiple files in the same project reuse the same client."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    file1 = project / "File1.lean"
    file1.write_text("theorem a : True := by trivial")

    file2 = project / "src" / "File2.lean"
    file2.parent.mkdir(parents=True)
    file2.write_text("theorem b : True := by trivial")

    ctx = _Context(_LifespanContext(None, None))

    # Setup for first file
    rel_path1 = setup_client_for_file(ctx, str(file1))
    assert rel_path1 == "File1.lean"
    first_client = ctx.request_context.lifespan_context.client
    assert len(patched_clients) == 1

    # Setup for second file in same project should reuse client
    rel_path2 = setup_client_for_file(ctx, str(file2))
    assert rel_path2 == "src/File2.lean"
    assert ctx.request_context.lifespan_context.client is first_client
    assert not first_client.closed
    assert len(patched_clients) == 1  # No new client created


def test_setup_client_for_file_switches_projects(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    """Verify that switching to a different project closes old client and creates new one."""
    project1 = tmp_path / "proj1"
    project1.mkdir()
    (project1 / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    file1 = project1 / "File1.lean"
    file1.write_text("theorem a : True := by trivial")

    project2 = tmp_path / "proj2"
    project2.mkdir()
    (project2 / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    file2 = project2 / "File2.lean"
    file2.write_text("theorem b : True := by trivial")

    ctx = _Context(_LifespanContext(None, None))

    # Setup for first project
    rel_path1 = setup_client_for_file(ctx, str(file1))
    assert rel_path1 == "File1.lean"
    first_client = ctx.request_context.lifespan_context.client
    assert len(patched_clients) == 1

    # Switch to second project
    rel_path2 = setup_client_for_file(ctx, str(file2))
    assert rel_path2 == "File2.lean"
    second_client = ctx.request_context.lifespan_context.client

    # Old client should be closed, new one created
    assert first_client.closed
    assert second_client is not first_client
    assert len(patched_clients) == 2
    assert ctx.request_context.lifespan_context.lean_project_path == project2


def test_startup_client_serializes_concurrent_calls(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    ctx = _Context(_LifespanContext(project, None))

    def _invoke_startup() -> None:
        startup_client(ctx)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_invoke_startup) for _ in range(10)]
        for future in futures:
            assert future.result() is None

    assert len(patched_clients) == 1
    assert ctx.request_context.lifespan_context.client is patched_clients[0]


def test_setup_client_for_file_forces_reopen_on_import_change(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    lean_file = project / "Foo.lean"
    lean_file.write_text("import Mathlib\n\n#check Nat", encoding="utf-8")

    ctx = _Context(_LifespanContext(None, None))

    rel_path = setup_client_for_file(ctx, str(lean_file))
    assert rel_path == "Foo.lean"

    client = patched_clients[0]
    assert client.open_calls == [("Foo.lean", False)]
    assert client.update_calls == []

    # Change imports: this should force reopen instead of plain didChange update.
    lean_file.write_text("import Std\n\n#check Nat", encoding="utf-8")
    rel_path_again = setup_client_for_file(ctx, str(lean_file))
    assert rel_path_again == "Foo.lean"

    assert ("Foo.lean", True) in client.open_calls
    assert client.update_calls == []

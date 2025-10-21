from __future__ import annotations

from __future__ import annotations

from pathlib import Path

import pytest

from lean_lsp_mcp.client_utils import (
    setup_client_for_file,
    startup_client,
    valid_lean_project_path,
)


class _MockLeanClient:
    def __init__(self, project_path: str) -> None:
        self.project_path = project_path
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _LifespanContext:
    def __init__(
        self, lean_project_path: str | None, client: _MockLeanClient | None
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
        project_path: str, initial_build: bool
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

    ctx = _Context(_LifespanContext(str(project), None))

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
    ctx.request_context.lifespan_context.lean_project_path = str(new_project)

    startup_client(ctx)
    assert first.closed
    assert ctx.request_context.lifespan_context.client.project_path == str(new_project)
    assert len(patched_clients) == 2


def test_valid_lean_project_path(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0")

    assert valid_lean_project_path(str(project))
    assert not valid_lean_project_path(str(project / "missing"))


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
    assert ctx.request_context.lifespan_context.client.project_path == str(project)
    assert len(patched_clients) == 1

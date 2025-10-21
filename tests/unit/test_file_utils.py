from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lean_lsp_mcp.file_utils import (
    get_file_contents,
    get_relative_file_path,
    update_file,
)


class _FakeClient:
    def __init__(self) -> None:
        self.closed: list[list[str]] = []

    def close_files(self, files: list[str]) -> None:
        self.closed.append(files)


class _LifespanContext:
    def __init__(self, project_path: Path, client: _FakeClient) -> None:
        self.lean_project_path = project_path
        self.client = client
        self.file_content_hashes: dict[str, int] = {}


class _RequestContext:
    def __init__(self, lifespan_context: _LifespanContext) -> None:
        self.lifespan_context = lifespan_context


class _Context:
    def __init__(self, lifespan_context: _LifespanContext) -> None:
        self.request_context = _RequestContext(lifespan_context)


def test_get_relative_file_path_handles_absolute_and_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path
    target = project / "src" / "Example.lean"
    target.parent.mkdir(parents=True)
    target.write_text("example")

    # absolute
    assert get_relative_file_path(project, str(target)) == "src/Example.lean"

    # relative to project
    assert (
        get_relative_file_path(project, "src/Example.lean") == "src/Example.lean"
    )

    # relative to CWD
    monkeypatch.chdir(project)
    assert (
        get_relative_file_path(project, "src/Example.lean") == "src/Example.lean"
    )


def test_get_file_contents_fallback_encoding(tmp_path: Path) -> None:
    latin1_file = tmp_path / "latin1.txt"
    latin1_file.write_text("caf\xe9", encoding="latin-1")

    assert get_file_contents(str(latin1_file)) == "caf\xe9"


def test_update_file_tracks_changes(tmp_path: Path) -> None:
    project = tmp_path
    lean_file = project / "Main.lean"
    lean_file.write_text("initial")

    client = _FakeClient()
    lifespan = _LifespanContext(project, client)
    ctx = _Context(lifespan)

    # first read populates cache but does not close files
    contents = update_file(ctx, "Main.lean")
    assert contents == "initial"
    assert client.closed == []

    # unchanged content still avoids close
    contents = update_file(ctx, "Main.lean")
    assert contents == "initial"
    assert client.closed == []

    # mutated file triggers close
    lean_file.write_text("updated")
    contents = update_file(ctx, "Main.lean")
    assert contents == "updated"
    assert client.closed == [["Main.lean"]]

    # cache now reflects latest version
    assert lifespan.file_content_hashes["Main.lean"] == hash("updated")

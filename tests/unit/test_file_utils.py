from __future__ import annotations

from pathlib import Path

import pytest

from lean_lsp_mcp.file_utils import (
    get_file_contents,
    get_relative_file_path,
)


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
    assert get_relative_file_path(project, "src/Example.lean") == "src/Example.lean"

    # relative to CWD
    monkeypatch.chdir(project)
    assert get_relative_file_path(project, "src/Example.lean") == "src/Example.lean"


def test_get_file_contents_fallback_encoding(tmp_path: Path) -> None:
    latin1_file = tmp_path / "latin1.txt"
    latin1_file.write_text("caf\xe9", encoding="latin-1")

    assert get_file_contents(str(latin1_file)) == "caf\xe9"

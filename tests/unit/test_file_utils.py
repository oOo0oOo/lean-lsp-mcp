from __future__ import annotations

import os
from pathlib import Path

import pytest

from lean_lsp_mcp.file_utils import (
    build_lean_path_policy,
    get_file_contents,
    get_relative_file_path,
    require_lean_project_path,
    resolve_input_path,
)


def _make_project(root: Path) -> Path:
    root.mkdir()
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    (root / "lakefile.toml").write_text('name = "test"\n')
    return root


def test_get_relative_file_path_handles_absolute_and_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_project(tmp_path / "proj")
    target = project / "src" / "Example.lean"
    target.parent.mkdir(parents=True)
    target.write_text("example")

    assert get_relative_file_path(project, str(target)) == "src/Example.lean"
    assert get_relative_file_path(project, "src/Example.lean") == "src/Example.lean"

    monkeypatch.chdir(project)
    assert get_relative_file_path(project, "src/Example.lean") == "src/Example.lean"


def test_require_lean_project_path_requires_lakefile(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    with pytest.raises(ValueError, match="must contain `lean-toolchain`"):
        require_lean_project_path(project)


def test_build_lean_path_policy_allows_project_dependency_and_stdlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_project(tmp_path / "proj")

    dep_real_root = tmp_path / "deps" / "mathlib"
    dep_file = dep_real_root / "Mathlib" / "Algebra" / "Group.lean"
    dep_file.parent.mkdir(parents=True)
    dep_file.write_text("theorem depThm : True := by trivial\n")

    dep_link = project / ".lake" / "packages" / "mathlib"
    dep_link.parent.mkdir(parents=True)
    dep_link.symlink_to(dep_real_root, target_is_directory=True)

    stdlib_root = tmp_path / "lean-prefix" / "src"
    stdlib_file = stdlib_root / "Init" / "Prelude.lean"
    stdlib_file.parent.mkdir(parents=True)
    stdlib_file.write_text("def stdlibDef : Nat := 0\n")
    monkeypatch.setattr(
        "lean_lsp_mcp.file_utils._stdlib_src_root",
        lambda _project_root: stdlib_root,
    )

    policy = build_lean_path_policy(project)

    assert policy.validate_path(project / "lakefile.toml") == project / "lakefile.toml"
    assert (
        policy.display_path(dep_file)
        == ".lake/packages/mathlib/Mathlib/Algebra/Group.lean"
    )
    assert policy.display_path(stdlib_file) == ".lean-stdlib/Init/Prelude.lean"
    assert policy.client_relative_path(dep_file) == os.path.relpath(dep_file, project)
    assert policy.client_relative_path(stdlib_file) == os.path.relpath(
        stdlib_file, project
    )


def test_build_lean_path_policy_rejects_symlink_escape(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "proj")
    outside = tmp_path / "outside" / "Secrets.lean"
    outside.parent.mkdir(parents=True)
    outside.write_text("def leaked : Nat := 0\n")

    link_path = project / "src" / "Secrets.lean"
    link_path.parent.mkdir(parents=True)
    try:
        link_path.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"symlinks unavailable: {exc}")

    policy = build_lean_path_policy(project)
    resolved = resolve_input_path("src/Secrets.lean", project_root=project)

    with pytest.raises(ValueError, match="outside the active Lean project"):
        policy.validate_path(resolved)


def test_get_file_contents_fallback_encoding(tmp_path: Path) -> None:
    latin1_file = tmp_path / "latin1.txt"
    latin1_file.write_text("caf\xe9", encoding="latin-1")

    assert get_file_contents(latin1_file) == "caf\xe9"

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final, Sequence

__all__ = ["ensure_test_project"]

PROJECT_DIRNAME: Final[str] = "test_project"
LEAN_TOOLCHAIN: Final[str] = "leanprover/lean4:v4.25.0\n"

LAKEFILE_TOML: Final[str] = """name = \"McpTestProject\"
version = \"0.1.0\"
defaultTargets = [\"McpTestProject\"]

[[require]]
name = \"mathlib\"
scope = \"leanprover-community\"
rev = \"v4.25.0\"

[[lean_lib]]
name = \"McpTestProject\"
"""

LIB_MAIN_LEAN: Final[str] = """import Mathlib

abbrev sampleValue : ℕ := 42
"""

LAKE_UPDATE: Final[Sequence[str]] = ("lake", "update", "--keep-toolchain")
LAKE_BUILD_STEPS: Final[tuple[Sequence[str], ...]] = (
    ("lake", "exe", "cache", "get"),
    ("lake", "build"),
)


def ensure_test_project(repo_root: Path) -> Path:
    project_root = repo_root / "tests" / PROJECT_DIRNAME
    project_root.mkdir(parents=True, exist_ok=True)

    _write_if_changed(project_root / "lean-toolchain", LEAN_TOOLCHAIN)
    _write_if_changed(project_root / "lakefile.toml", LAKEFILE_TOML)
    _write_if_changed(project_root / "McpTestProject.lean", LIB_MAIN_LEAN)

    should_run_setup = _should_refresh(project_root)

    if should_run_setup:
        _run_lake_steps(project_root)

    return project_root


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def _should_refresh(project_root: Path) -> bool:
    mathlib_dir = project_root / ".lake" / "packages" / "mathlib"
    olean_dir = project_root / ".lake" / "build"
    return not mathlib_dir.exists() or not olean_dir.exists()


def _run_lake_steps(project_root: Path) -> None:
    manifest_path = project_root / "lake-manifest.json"
    if not manifest_path.exists():
        try:
            subprocess.run(LAKE_UPDATE, cwd=project_root, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "`lake` executable is required for end-to-end tests"
            ) from exc

    for args in LAKE_BUILD_STEPS:
        try:
            subprocess.run(args, cwd=project_root, check=True)
        except (
            FileNotFoundError
        ) as exc:  # pragma: no cover - relies on user environment
            raise RuntimeError(
                "`lake` executable is required for end-to-end tests"
            ) from exc

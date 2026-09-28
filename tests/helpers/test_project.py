from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final, Sequence

__all__ = ["ensure_test_project"]

PROJECT_DIRNAME: Final[str] = "test_project"
LEAN_TOOLCHAIN: Final[str] = "leanprover/lean4:v4.34.1\n"

LAKEFILE_TOML: Final[str] = """name = \"McpTestProject\"
version = \"0.1.0\"
defaultTargets = [\"McpTestProject\"]

[[require]]
name = \"mathlib\"
scope = \"leanprover-community\"
rev = \"v4.34.1\"

[[require]]
name = \"REPL\"
git = \"https://github.com/leanprover-community/repl\"
rev = \"v4.34.0\"

[[lean_lib]]
name = \"McpTestProject\"

[[lean_lib]]
name = \"VerifyTest\"
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
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _should_refresh(project_root: Path) -> bool:
    mathlib_dir = project_root / ".lake" / "packages" / "mathlib"
    olean_dir = project_root / ".lake" / "build"
    if not mathlib_dir.exists() or not olean_dir.exists():
        return True
    # A checkout can retain .lake from the previous test-project toolchain.
    # Directory existence alone does not make those cached artifacts usable.
    toolchain = mathlib_dir / "lean-toolchain"
    return (
        not toolchain.is_file()
        or toolchain.read_text(encoding="utf-8").strip() != LEAN_TOOLCHAIN.strip()
    )


def _run_lake_steps(project_root: Path) -> None:
    # Even with a checked-in manifest, existing package checkouts may still
    # belong to the old toolchain. Update them before fetching build artifacts.
    for args in (LAKE_UPDATE, *LAKE_BUILD_STEPS):
        try:
            subprocess.run(args, cwd=project_root, check=True)
        except (
            FileNotFoundError
        ) as exc:  # pragma: no cover - relies on user environment
            raise RuntimeError(
                "`lake` executable is required for end-to-end tests"
            ) from exc

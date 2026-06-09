"""Build the Nix flake end to end.

The flake derives its whole dependency closure from `uv.lock` via uv2nix, so a
`uv lock` bump flows through automatically. This test builds it and runs the
resulting binary to catch breakage in that pipeline (e.g. a dependency that
needs a build-system override).

Skipped automatically when `nix` is not installed, so it is a no-op for
contributors and CI without Nix. It is also marked `slow` (a clean build pulls
the dependency closure), so the default `-m "not slow"` run skips it; run it
explicitly with a full `pytest tests/`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_NIX = shutil.which("nix")
_HAS_FLAKE = (REPO_ROOT / "flake.nix").exists()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _NIX is None or not _HAS_FLAKE,
        reason="nix not installed or flake.nix missing",
    ),
]

_NIX_FLAGS = ["--extra-experimental-features", "nix-command flakes"]


def test_nix_flake_builds_and_runs() -> None:
    build = subprocess.run(
        [_NIX, "build", *_NIX_FLAGS, ".#default", "--no-link", "--print-out-paths"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert build.returncode == 0, f"nix build failed:\n{build.stderr}"

    out_path = Path(build.stdout.strip().splitlines()[-1])
    binary = out_path / "bin" / "lean-lsp-mcp"
    assert binary.exists(), f"expected binary at {binary}"

    run = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, f"binary --version failed:\n{run.stderr}"
    assert "lean-lsp-mcp" in run.stdout

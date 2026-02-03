#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.helpers.mcp_client import MCPClient, connect_stdio_client
from tests.helpers.test_project import ensure_test_project


def _default_bench_file() -> str:
    return (
        textwrap.dedent(
            """
        import Mathlib

        theorem bench_add_comm (n m : Nat) : n + m = m + n := by
          sorry
        """
        ).strip()
        + "\n"
    )


def _ensure_bench_file(project_path: Path) -> tuple[Path, int]:
    path = project_path / "Bench.lean"
    content = _default_bench_file()
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    # Line numbers are 1-indexed; snippet goes on the `sorry` line.
    return path, 4


def _server_env(
    repo_root: Path,
    project_path: Path,
    *,
    use_repl: bool,
    repl_path: Path | None,
    workers: int,
    timeout: int,
) -> dict[str, str]:
    pythonpath_entries = [str(repo_root / "src")]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)

    env: dict[str, str] = {
        "PYTHONPATH": os.pathsep.join(pythonpath_entries),
        "LEAN_PROJECT_PATH": str(project_path),
        "LEAN_LOG_LEVEL": os.environ.get("LEAN_LOG_LEVEL", "ERROR"),
        "LEAN_LSP_TEST_MODE": "1",
    }

    token = os.environ.get("LEAN_LSP_MCP_TOKEN")
    if token:
        env["LEAN_LSP_MCP_TOKEN"] = token

    if use_repl:
        if repl_path is None:
            raise RuntimeError("REPL enabled but no repl path provided")
        env.update(
            {
                "LEAN_REPL": "1",
                "LEAN_REPL_PATH": str(repl_path),
                "LEAN_REPL_WORKERS": str(workers),
                "LEAN_REPL_TIMEOUT": str(timeout),
            }
        )

    return env


def _find_repl_binary(repo_root: Path, project_path: Path) -> Path | None:
    candidates = [
        project_path / ".lake" / "build" / "bin" / "repl",
        repo_root / "repl" / ".lake" / "build" / "bin" / "repl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    which = shutil.which("repl")
    return Path(which) if which else None


def _summarize(times: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }


async def _run_multi_attempt(
    client: MCPClient,
    *,
    file_path: Path,
    line: int,
    snippets: Sequence[str],
) -> None:
    await client.call_tool(
        "lean_multi_attempt",
        {
            "file_path": str(file_path),
            "line": line,
            "snippets": list(snippets),
        },
    )


async def _bench_mode(
    repo_root: Path,
    env: dict[str, str],
    *,
    file_path: Path,
    line: int,
    snippets: Sequence[str],
    iterations: int,
    warmup: int,
) -> tuple[list[float], bool]:
    async with connect_stdio_client(
        sys.executable,
        ["-m", "lean_lsp_mcp", "--transport", "stdio"],
        env=env,
        cwd=repo_root,
    ) as client:
        for _ in range(warmup):
            await _run_multi_attempt(
                client,
                file_path=file_path,
                line=line,
                snippets=snippets,
            )

        times: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            await _run_multi_attempt(
                client,
                file_path=file_path,
                line=line,
                snippets=snippets,
            )
            times.append(time.perf_counter() - start)

        return times, False


def _print_env_info(project_path: Path) -> None:
    def _run(cmd: Sequence[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=project_path, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    lean_version = _run(["lake", "env", "lean", "--version"])
    lake_version = _run(["lake", "--version"])
    print(f"Lean: {lean_version}")
    print(f"Lake: {lake_version}")


def _print_summary(label: str, times: Sequence[float]) -> None:
    stats = _summarize(times)
    print(
        f"{label}: mean={stats['mean']:.3f}s "
        f"median={stats['median']:.3f}s min={stats['min']:.3f}s max={stats['max']:.3f}s"
    )


def _iter_snippets() -> Iterable[str]:
    return [
        "  simp",
        "  exact Nat.add_comm _ _",
        "  sorry",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark lean_multi_attempt with and without REPL pooling."
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--repl-path", type=Path)
    parser.add_argument("--skip-repl", action="store_true")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    project_path = ensure_test_project(repo_root)
    bench_file, line = _ensure_bench_file(project_path)

    print("Project:", project_path)
    print("File:", bench_file)
    _print_env_info(project_path)

    repl_path = args.repl_path or _find_repl_binary(repo_root, project_path)

    snippets = list(_iter_snippets())
    print(f"Snippets ({len(snippets)}): {snippets}")

    env_no_repl = _server_env(
        repo_root,
        project_path,
        use_repl=False,
        repl_path=None,
        workers=args.workers,
        timeout=args.timeout,
    )

    no_repl_times, _ = asyncio.run(
        _bench_mode(
            repo_root,
            env_no_repl,
            file_path=bench_file,
            line=line,
            snippets=snippets,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    )
    _print_summary("LSP (no REPL)", no_repl_times)

    if args.skip_repl:
        return 0

    if repl_path is None or not repl_path.exists():
        print("REPL binary not found; skipping REPL benchmark.")
        return 0

    env_repl = _server_env(
        repo_root,
        project_path,
        use_repl=True,
        repl_path=repl_path,
        workers=args.workers,
        timeout=args.timeout,
    )

    repl_times, _ = asyncio.run(
        _bench_mode(
            repo_root,
            env_repl,
            file_path=bench_file,
            line=line,
            snippets=snippets,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    )
    _print_summary("REPL pool", repl_times)

    speedup = statistics.mean(no_repl_times) / statistics.mean(repl_times)
    print(f"Speedup (mean): {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

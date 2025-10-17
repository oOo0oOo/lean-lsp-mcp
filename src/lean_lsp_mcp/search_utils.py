"""Utilities for Lean search tools."""

from __future__ import annotations

from collections.abc import Iterable
import platform
import re
import shutil
import subprocess
from orjson import loads as _json_loads
from pathlib import Path


INSTALL_URL = "https://github.com/BurntSushi/ripgrep#installation"

_PLATFORM_INSTRUCTIONS: dict[str, Iterable[str]] = {
    "Windows": (
        "winget install BurntSushi.ripgrep.MSVC",
        "choco install ripgrep",
    ),
    "Darwin": ("brew install ripgrep",),
    "Linux": (
        "sudo apt-get install ripgrep",
        "sudo dnf install ripgrep",
    ),
}


def check_ripgrep_status() -> tuple[bool, str]:
    """Check whether ``rg`` is available on PATH and return status + message."""

    if shutil.which("rg"):
        return True, ""

    system = platform.system()
    platform_instructions = _PLATFORM_INSTRUCTIONS.get(system, ("Check alternative installation methods.",))

    lines = [
        "ripgrep (rg) was not found on your PATH. The lean_local_search tool uses ripgrep for fast declaration search.",
        "",
        "Installation options:",
        *(f"  - {item}" for item in platform_instructions),
        f"More installation options: {INSTALL_URL}",
    ]

    return False, "\n".join(lines)


def lean_search(
    query: str,
    limit: int = 100,
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    """Search Lean declarations matching ``query`` using ripgrep."""
    root = (project_root or Path.cwd()).resolve()
    escaped_query = re.escape(query)
    ripgrep_pattern = (
        rf"^\s*(?:theorem|lemma|def|axiom|class|instance|structure|inductive|abbrev|opaque)\s+"
        rf"{escaped_query}[A-Za-z0-9_'.]*(?:\s|:)"
    )

    command = [
        "rg",
        "--json",
        "--no-ignore",
        "--smart-case",
        "--hidden",
        "--color",
        "never",
        "-g",
        "*.lean",
        "-g",
        "!.git/**",
        "-g",
        "!.lake/build/**",
        ripgrep_pattern,
        "."
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(root)
    )

    results: list[dict[str, str]] = []

    for raw_line in completed.stdout.splitlines():
        if not raw_line:
            continue

        event = _json_loads(raw_line)

        if event.get("type") != "match":
            continue

        data = event["data"]
        line_text = data["lines"]["text"]
        parts = line_text.lstrip().split(maxsplit=2)
        if len(parts) < 2:
            continue

        decl_kind, raw_name = parts[0], parts[1]
        decl_name = raw_name.rstrip(":")

        path_text = data["path"]["text"]
        file_path = Path(path_text)
        absolute_path = file_path if file_path.is_absolute() else (root / file_path).resolve()
        try:
            display_path = str(absolute_path.relative_to(root))
        except ValueError:
            display_path = str(file_path)

        results.append({"name": decl_name, "kind": decl_kind, "file": display_path})

        if len(results) >= limit:
            break

    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"ripgrep exited with code {completed.returncode}\n{completed.stderr}"
        )

    return results

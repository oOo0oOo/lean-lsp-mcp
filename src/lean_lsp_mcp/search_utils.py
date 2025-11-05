"""Utilities for Lean search tools."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
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
    platform_instructions = _PLATFORM_INSTRUCTIONS.get(
        system, ("Check alternative installation methods.",)
    )

    lines = [
        "ripgrep (rg) was not found on your PATH. The lean_local_search tool uses ripgrep for fast declaration search.",
        "",
        "Installation options:",
        *(f"  - {item}" for item in platform_instructions),
        f"More installation options: {INSTALL_URL}",
    ]

    return False, "\n".join(lines)


def lean_local_search(
    query: str,
    limit: int = 32,
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    """Search Lean declarations matching ``query`` using ripgrep; results include theorems, lemmas, defs, classes, instances, structures, inductives, abbrevs, and opaque decls."""
    root = (project_root or Path.cwd()).resolve()

    pattern = (
        rf"^\s*(?:theorem|lemma|def|axiom|class|instance|structure|inductive|abbrev|opaque)\s+"
        rf"(?:[A-Za-z0-9_'.]+\.)*{re.escape(query)}[A-Za-z0-9_'.]*(?:\s|:)"
    )

    command = [
        "rg",
        "--json",
        "--no-ignore",
        "--smart-case",
        "--hidden",
        "--color",
        "never",
        "--no-messages",
        "-g",
        "*.lean",
        "-g",
        "!.git/**",
        "-g",
        "!.lake/build/**",
        pattern,
        str(root),
    ]

    if lean_src := _get_lean_src_search_path():
        command.append(lean_src)

    result = subprocess.run(command, capture_output=True, text=True, cwd=str(root))

    matches = []
    for line in result.stdout.splitlines():
        if not line or (event := _json_loads(line)).get("type") != "match":
            continue

        data = event["data"]
        parts = data["lines"]["text"].lstrip().split(maxsplit=2)
        if len(parts) < 2:
            continue

        decl_kind, decl_name = parts[0], parts[1].rstrip(":")
        file_path = Path(data["path"]["text"])
        abs_path = (
            file_path if file_path.is_absolute() else (root / file_path).resolve()
        )

        try:
            display_path = str(abs_path.relative_to(root))
        except ValueError:
            display_path = str(file_path)

        matches.append({"name": decl_name, "kind": decl_kind, "file": display_path})

        if len(matches) >= limit:
            break

    if result.returncode not in (0, 1) and not matches:
        error_msg = f"ripgrep exited with code {result.returncode}"
        if result.stderr:
            error_msg += f"\n{result.stderr}"
        raise RuntimeError(error_msg)

    return matches


@lru_cache(maxsize=1)
def _get_lean_src_search_path() -> str | None:
    """Return the Lean stdlib directory, if available (cache once)."""
    try:
        completed = subprocess.run(
            ["lean", "--print-prefix"], capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    prefix = completed.stdout.strip()
    if not prefix:
        return None

    candidate = Path(prefix).expanduser().resolve() / "src"
    if candidate.exists():
        return str(candidate)

    return None

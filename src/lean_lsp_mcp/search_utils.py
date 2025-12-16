"""Utilities for Lean search tools."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
import platform
import re
import shutil
import subprocess
import threading
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


def _create_ripgrep_process(command: list[str], *, cwd: str) -> subprocess.Popen[str]:
    """Spawn ripgrep and return a process with line-streaming stdout.

    Separated for test monkeypatching and to allow early termination once we
    have enough matches.
    """
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )


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

    process = _create_ripgrep_process(command, cwd=str(root))

    matches: list[dict[str, str]] = []
    stderr_text = ""
    terminated_early = False
    stderr_chunks: list[str] = []
    stderr_chars = 0
    stderr_truncated = False
    max_stderr_chars = 100_000

    def _drain_stderr(pipe) -> None:
        nonlocal stderr_chars, stderr_truncated
        try:
            for err_line in pipe:
                if stderr_chars < max_stderr_chars:
                    stderr_chunks.append(err_line)
                    stderr_chars += len(err_line)
                else:
                    stderr_truncated = True
        except Exception:
            return

    stderr_thread: threading.Thread | None = None
    if process.stderr is not None:
        stderr_thread = threading.Thread(
            target=_drain_stderr,
            args=(process.stderr,),
            name="lean-local-search-rg-stderr",
            daemon=True,
        )
        stderr_thread.start()

    try:
        stdout = process.stdout
        if stdout is None:
            raise RuntimeError("ripgrep did not provide stdout pipe")

        for line in stdout:
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
                terminated_early = True
                try:
                    process.terminate()
                except Exception:
                    pass
                break

        try:
            if terminated_early:
                process.wait(timeout=5)
            else:
                process.wait()
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        if process.returncode is None:
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
        if stderr_thread is not None:
            stderr_thread.join(timeout=1)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    if stderr_chunks:
        stderr_text = "".join(stderr_chunks)
        if stderr_truncated:
            stderr_text += "\n[stderr truncated]"

    returncode = process.returncode if process.returncode is not None else 0

    if returncode not in (0, 1) and not matches:
        error_msg = f"ripgrep exited with code {returncode}"
        if stderr_text:
            error_msg += f"\n{stderr_text}"
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

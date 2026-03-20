"""Utilities for Lean search tools."""

from __future__ import annotations

from collections.abc import Iterable
import platform
import re
import shutil
import subprocess
import threading
from orjson import loads as _json_loads
from pathlib import Path

from lean_lsp_mcp.file_utils import LeanPathPolicy, build_lean_path_policy

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
    try:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=cwd,
        )
    except FileNotFoundError:
        _ok, msg = check_ripgrep_status()
        if not _ok:
            raise FileNotFoundError(msg) from None
        raise


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


def _local_search_sort_key(
    match: dict[str, str], normalized_query: str
) -> tuple[int, int, int, str, str]:
    """Sort local search results by relevance and stability.

    Priorities:
    1. Exact declaration-name match over prefixes/suffixes.
    2. Project declarations over `.lake/packages` dependencies.
    3. Shorter base names, then lexical fallback for deterministic order.
    """
    name = match["name"]
    basename = name.rsplit(".", 1)[-1]
    name_fold = name.casefold()
    base_fold = basename.casefold()

    if "." in normalized_query:
        if name_fold == normalized_query:
            relevance_rank = 0
        elif name_fold.startswith(normalized_query):
            relevance_rank = 1
        elif normalized_query in name_fold:
            relevance_rank = 2
        elif base_fold == normalized_query:
            relevance_rank = 3
        elif base_fold.startswith(normalized_query):
            relevance_rank = 4
        elif normalized_query in base_fold:
            relevance_rank = 5
        else:
            relevance_rank = 6
    else:
        if name_fold == normalized_query or base_fold == normalized_query:
            relevance_rank = 0
        elif base_fold.startswith(normalized_query):
            relevance_rank = 1
        elif normalized_query in base_fold:
            relevance_rank = 2
        elif name_fold.startswith(normalized_query):
            relevance_rank = 3
        elif normalized_query in name_fold:
            relevance_rank = 4
        else:
            relevance_rank = 5

    package_penalty = 1 if match["file"].startswith(".lake/packages/") else 0
    return (relevance_rank, package_penalty, len(basename), basename, name)


def _resolve_namespaces(file_path: Path, line_numbers: set[int]) -> dict[int, str]:
    """Return the enclosing namespace prefix for each 1-indexed *line_number*."""
    if not line_numbers:
        return {}
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return {}

    scope_stack: list[str | None] = []  # None = section/mutual (not part of FQN)
    result: dict[int, str] = {}

    for i, raw in enumerate(lines[: max(line_numbers)], 1):
        stripped = raw.strip()
        if m := re.match(r"^namespace\s+([\w.']+)", stripped):
            scope_stack.append(m.group(1))
        elif re.match(r"^(?:section|mutual)\b", stripped):
            scope_stack.append(None)
        elif re.match(r"^end\b", stripped):
            if scope_stack:
                scope_stack.pop()

        if i in line_numbers:
            result[i] = ".".join(s for s in scope_stack if s is not None)

    return result


def lean_local_search(
    query: str,
    limit: int = 32,
    project_root: Path | None = None,
    path_policy: LeanPathPolicy | None = None,
) -> list[dict[str, str]]:
    """Search Lean declarations matching ``query`` using ripgrep; results include theorems, lemmas, defs, classes, instances, structures, inductives, abbrevs, and opaque decls."""
    policy = path_policy
    if policy is None:
        root = (project_root or Path.cwd()).resolve()
        policy = build_lean_path_policy(root)
    root = policy.project_root

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

    if policy.stdlib_root is not None:
        command.append(str(policy.stdlib_root))

    process = _create_ripgrep_process(command, cwd=str(root))

    matches: list[dict[str, str]] = []
    match_locations: list[tuple[Path, int]] = []
    max_candidates = min(max(limit * 8, limit), 2048)
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
            line_number = data.get("line_number", 0)
            file_path = Path(data["path"]["text"])
            abs_path = (
                file_path if file_path.is_absolute() else (root / file_path).resolve()
            )

            try:
                display_path = policy.display_path(abs_path)
            except ValueError:
                continue

            matches.append({"name": decl_name, "kind": decl_kind, "file": display_path})
            match_locations.append((abs_path, line_number))

            if len(matches) >= max_candidates:
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

    # Resolve enclosing namespaces and qualify declaration names.
    file_lines: dict[Path, set[int]] = {}
    for abs_path, line_num in match_locations:
        file_lines.setdefault(abs_path, set()).add(line_num)
    ns_cache: dict[Path, dict[int, str]] = {
        fp: _resolve_namespaces(fp, lns) for fp, lns in file_lines.items()
    }
    for match, (abs_path, line_num) in zip(matches, match_locations):
        prefix = ns_cache.get(abs_path, {}).get(line_num, "")
        if prefix:
            match["name"] = f"{prefix}.{match['name']}"

    normalized_query = query.casefold()
    matches.sort(key=lambda match: _local_search_sort_key(match, normalized_query))

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in matches:
        key = (match["name"], match["kind"], match["file"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
        if len(deduped) >= limit:
            break

    return deduped

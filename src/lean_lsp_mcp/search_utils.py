"""Utilities for Lean search tools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import platform
import re
import shutil
import subprocess
import threading
from orjson import loads as _json_loads
from pathlib import Path

from lean_lsp_mcp.file_utils import LeanPathPolicy, build_lean_path_policy

INSTALL_URL = "https://github.com/BurntSushi/ripgrep#installation"

# Declaration keywords and the attributes/modifiers that may precede them.
# The search pattern and the line parser are built from the same building
# blocks so they can never drift out of sync.
_DECL_KEYWORDS = (
    "theorem",
    "lemma",
    "def",
    "axiom",
    "class",
    "instance",
    "structure",
    "inductive",
    "abbrev",
    "opaque",
)
_DECL_MODIFIERS = (
    "public",
    "protected",
    "private",
    "noncomputable",
    "partial",
    "unsafe",
    "scoped",
    "local",
)
# Leading `@[...]` attributes and zero or more modifiers before the keyword.
_DECL_LEAD = (
    r"^\s*(?:@\[[^\]]*\]\s*)*"
    r"(?:(?:" + "|".join(_DECL_MODIFIERS) + r")\s+)*"
)
_DECL_KEYWORD_ALT = "|".join(_DECL_KEYWORDS)
# Parses a matched line into its declaration keyword and (possibly dotted) name,
# skipping any attributes/modifiers that precede the keyword.
_DECL_LINE_RE = re.compile(
    _DECL_LEAD
    + rf"(?P<kind>{_DECL_KEYWORD_ALT})\s+(?P<name>[A-Za-z0-9_']+(?:\.[A-Za-z0-9_']+)*)"
)

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

_MAX_RIPGREP_STDERR_CHARS = 100_000


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


@dataclass
class _StderrCapture:
    chunks: list[str] = field(default_factory=list)
    chars: int = 0
    truncated: bool = False

    def drain(self, pipe: Iterable[str]) -> None:
        try:
            for line in pipe:
                if self.chars < _MAX_RIPGREP_STDERR_CHARS:
                    self.chunks.append(line)
                    self.chars += len(line)
                else:
                    self.truncated = True
        except Exception:
            return

    def text(self) -> str:
        output = "".join(self.chunks)
        if output and self.truncated:
            output += "\n[stderr truncated]"
        return output


def _read_ripgrep_matches(
    process: subprocess.Popen[str],
    root: Path,
    policy: LeanPathPolicy,
    max_candidates: int,
) -> tuple[list[dict[str, str]], list[tuple[Path, int]], bool]:
    stdout = process.stdout
    if stdout is None:
        raise RuntimeError("ripgrep did not provide stdout pipe")

    matches: list[dict[str, str]] = []
    locations: list[tuple[Path, int]] = []
    for line in stdout:
        if not line or (event := _json_loads(line)).get("type") != "match":
            continue

        data = event["data"]
        declaration = _DECL_LINE_RE.match(data["lines"]["text"])
        if declaration is None:
            continue

        file_path = Path(data["path"]["text"])
        abs_path = (
            file_path if file_path.is_absolute() else (root / file_path).resolve()
        )
        try:
            display_path = policy.display_path(abs_path)
        except ValueError:
            continue

        matches.append(
            {
                "name": declaration.group("name"),
                "kind": declaration.group("kind"),
                "file": display_path,
            }
        )
        locations.append((abs_path, data.get("line_number", 0)))
        if len(matches) >= max_candidates:
            try:
                process.terminate()
            except Exception:
                pass
            return matches, locations, True

    return matches, locations, False


def _wait_for_ripgrep(
    process: subprocess.Popen[str], *, terminated_early: bool
) -> None:
    try:
        process.wait(timeout=5) if terminated_early else process.wait()
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _close_ripgrep_process(
    process: subprocess.Popen[str], stderr_thread: threading.Thread | None
) -> None:
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


def _qualify_and_rank_matches(
    matches: list[dict[str, str]],
    locations: list[tuple[Path, int]],
    query: str,
    limit: int,
) -> list[dict[str, str]]:
    file_lines: dict[Path, set[int]] = {}
    for abs_path, line_num in locations:
        file_lines.setdefault(abs_path, set()).add(line_num)
    namespaces = {
        path: _resolve_namespaces(path, lines) for path, lines in file_lines.items()
    }
    for match, (abs_path, line_num) in zip(matches, locations):
        prefix = namespaces.get(abs_path, {}).get(line_num, "")
        if prefix:
            match["name"] = f"{prefix}.{match['name']}"

    normalized_query = query.casefold()
    matches.sort(key=lambda match: _local_search_sort_key(match, normalized_query))

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in matches:
        key = (match["name"], match["kind"], match["file"])
        if key not in seen:
            seen.add(key)
            deduped.append(match)
        if len(deduped) >= limit:
            break
    return deduped


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
        # Optional attributes (`@[simp]`) and modifiers (`protected`, `private`,
        # `noncomputable`, ...) may precede the declaration keyword.
        _DECL_LEAD
        + rf"(?:{_DECL_KEYWORD_ALT})\s+"
        + rf"(?:[A-Za-z0-9_'.]+\.)*{re.escape(query)}[A-Za-z0-9_'.]*(?:\s|:)"
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
    max_candidates = min(max(limit * 8, limit), 2048)
    stderr = _StderrCapture()
    stderr_thread: threading.Thread | None = None
    if process.stderr is not None:
        stderr_thread = threading.Thread(
            target=stderr.drain,
            args=(process.stderr,),
            name="lean-local-search-rg-stderr",
            daemon=True,
        )
        stderr_thread.start()

    try:
        matches, locations, terminated_early = _read_ripgrep_matches(
            process, root, policy, max_candidates
        )
        _wait_for_ripgrep(process, terminated_early=terminated_early)
    finally:
        _close_ripgrep_process(process, stderr_thread)

    returncode = process.returncode if process.returncode is not None else 0

    if returncode not in (0, 1) and not matches:
        error_msg = f"ripgrep exited with code {returncode}"
        if stderr_text := stderr.text():
            error_msg += f"\n{stderr_text}"
        raise RuntimeError(error_msg)

    return _qualify_and_rank_matches(matches, locations, query, limit)


# `workspace/symbol` answers with LSP's generic SymbolKind enum, which does not
# carry Lean's declaration vocabulary (theorem / lemma / def / instance / ...).
# Reporting a neutral kind is honest; guessing one from the enum would not be.
INDEX_MATCH_KIND = "declaration"


def workspace_symbol_matches(
    symbols: Iterable[Mapping[str, object]],
    policy: LeanPathPolicy,
) -> list[dict[str, str]]:
    """Convert ``workspace/symbol`` results into local search match dicts.

    The language server answers from the compiled ``.ilean`` index, so it knows
    declarations that attributes such as ``@[to_additive]`` generate and that
    never exist as source text for ripgrep to find.

    Symbols outside the project, its dependencies and the stdlib are skipped so
    that ``file`` keeps the repo relative shape the ripgrep path produces.
    """
    matches: list[dict[str, str]] = []
    for symbol in symbols:
        name = symbol.get("name")
        if not isinstance(name, str) or not name:
            continue

        location = symbol.get("location")
        if not isinstance(location, Mapping):
            continue
        path = location.get("path")
        if not isinstance(path, str) or not path:
            continue

        local_path = Path(path)
        if not local_path.is_absolute():
            local_path = policy.project_root / local_path
        try:
            display_path = policy.display_path(local_path)
        except ValueError:
            continue

        matches.append({"name": name, "kind": INDEX_MATCH_KIND, "file": display_path})

    return matches


def merge_local_search_matches(
    source_matches: Sequence[dict[str, str]],
    index_matches: Sequence[dict[str, str]],
    query: str,
    limit: int,
) -> list[dict[str, str]]:
    """Combine ripgrep matches with index matches, the source entry winning ties.

    A declaration present in both is reported once, from the source match, which
    carries the real declaration keyword.

    Ranking reuses the ripgrep sort key so that a declaration found only in the
    index still outranks the near misses source search turned up: searching
    ``sum_range_succ`` puts the generated ``Finset.sum_range_succ`` ahead of
    ``Finset.sum_range_succ_mul_sum_range_succ``.
    """
    merged: list[dict[str, str]] = list(source_matches)
    seen_names = {match["name"] for match in merged}
    for match in index_matches:
        if match["name"] in seen_names:
            continue
        seen_names.add(match["name"])
        merged.append(match)

    normalized_query = query.casefold()
    merged.sort(key=lambda match: _local_search_sort_key(match, normalized_query))
    return merged[:limit]


@lru_cache(maxsize=4)
def _get_lean_src_search_path(project_root: Path | None = None) -> str | None:
    """Return the Lean stdlib directory, if available.

    Runs ``lean --print-prefix`` from *project_root* so that elan resolves the
    toolchain from the project's ``lean-toolchain`` file.
    """
    cwd = str(project_root) if project_root else None
    commands = [["lean", "--print-prefix"]]
    elan_lean = Path("~/.elan/bin/lean").expanduser()
    if elan_lean.exists():
        commands.append([str(elan_lean), "--print-prefix"])

    for command in commands:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=cwd,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

        prefix = completed.stdout.strip()
        if not prefix:
            continue

        candidate = Path(prefix).expanduser().resolve() / "src"
        if candidate.exists():
            return str(candidate)

    return None

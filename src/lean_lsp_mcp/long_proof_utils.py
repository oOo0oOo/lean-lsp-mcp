"""Detect long tactic proofs via ripgrep + file reading.

Follows the same pattern as search_utils.py and verify.py: ripgrep for fast
declaration finding, then read file content to measure proof blocks.
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path

from orjson import loads as _json_loads


# Regex for finding `:= by` within the first ~10 lines after a declaration
_PROOF_START_RE = re.compile(r":=\s*by\s*$|:=\s*by\s", re.MULTILINE)

# Declaration keyword pattern (same family as lean_local_search)
_DECL_PATTERN = (
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+|nonrec\s+|@\[.*?\]\s+)*"
    r"(?:theorem|lemma|def|instance)\s+(\S+)"
)

# Ripgrep pattern to find declaration start lines
_RG_DECL_PATTERN = r"^\s*(?:private\s+|protected\s+|noncomputable\s+|nonrec\s+)?(?:theorem|lemma|def|instance)\s"

_DECL_START_RE = re.compile(_DECL_PATTERN)


def _parse_rg_decl_line(text: str) -> tuple[str, str] | None:
    """Extract (keyword, name) from a declaration line. Returns None if no match."""
    m = _DECL_START_RE.match(text)
    if not m:
        return None
    name = m.group(1)
    # Extract keyword
    for kw in ("theorem", "lemma", "def", "instance"):
        if kw in text.split(name)[0]:
            return kw, name
    return None


def _find_declarations_rg(scan_path: Path) -> dict[Path, list[tuple[int, str, str]]]:
    """Find declarations via ripgrep. Returns {file: [(line_1indexed, keyword, name), ...]}."""
    command = [
        "rg",
        "--json",
        "--no-ignore",
        "--no-messages",
        "-g",
        "*.lean",
        "-g",
        "!.git/**",
        "-g",
        "!.lake/**",
        _RG_DECL_PATTERN,
    ]

    if scan_path.is_file():
        command.append(str(scan_path))
    else:
        command.append(str(scan_path))

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    results: dict[Path, list[tuple[int, str, str]]] = defaultdict(list)

    for line in proc.stdout.splitlines():
        if not line:
            continue
        event = _json_loads(line)
        if event.get("type") != "match":
            continue
        data = event["data"]
        text = data["lines"]["text"]
        line_num = data["line_number"]  # 1-indexed
        file_path = Path(data["path"]["text"])

        parsed = _parse_rg_decl_line(text)
        if not parsed:
            continue
        keyword, name = parsed

        # Handle anonymous instances
        if name == ":" or name.startswith(":"):
            name = f"instance@{line_num}"

        results[file_path].append((line_num, keyword, name))

    return dict(results)


def _measure_proofs(
    file_path: Path,
    decls: list[tuple[int, str, str]],
    root: Path,
) -> list[dict]:
    """Read a file and measure proof lengths for each declaration."""
    try:
        lines = file_path.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return []

    try:
        rel = str(file_path.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = str(file_path)

    entries = []
    # Sort by line number
    sorted_decls = sorted(decls, key=lambda d: d[0])

    for idx, (decl_line_1, keyword, name) in enumerate(sorted_decls):
        decl_line_0 = decl_line_1 - 1  # convert to 0-indexed

        # Find `:= by` within 10 lines
        proof_start = None
        search_end = min(decl_line_0 + 10, len(lines))
        for j in range(decl_line_0, search_end):
            if _PROOF_START_RE.search(lines[j]):
                proof_start = j
                break

        if proof_start is None:
            continue

        # Proof ends at next declaration or EOF
        if idx + 1 < len(sorted_decls):
            proof_end = sorted_decls[idx + 1][0] - 1  # 0-indexed
        else:
            proof_end = len(lines)

        # Trim trailing blank lines
        while proof_end > proof_start and not lines[proof_end - 1].strip():
            proof_end -= 1

        proof_lines = lines[proof_start:proof_end]
        line_count = len(proof_lines)

        have_count = sum(1 for ln in proof_lines if re.match(r"\s+have\b", ln))
        calc_count = sum(1 for ln in proof_lines if re.match(r"\s+calc\b", ln))

        entries.append(
            {
                "name": name,
                "kind": keyword,
                "file": rel,
                "line": decl_line_1,
                "line_count": line_count,
                "have_count": have_count,
                "calc_count": calc_count,
            }
        )

    return entries


def find_long_proofs(
    scan_path: Path,
    threshold: int = 30,
) -> tuple[list[dict], int]:
    """Find proofs exceeding the line threshold.

    Args:
        scan_path: File or directory to scan.
        threshold: Minimum proof lines to report.

    Returns:
        (entries, files_scanned) where entries are sorted by line_count descending.
    """
    root = scan_path if scan_path.is_dir() else scan_path.parent

    decls_by_file = _find_declarations_rg(scan_path)
    files_scanned = len(decls_by_file)

    all_entries = []
    for file_path, decls in decls_by_file.items():
        entries = _measure_proofs(file_path, decls, root)
        all_entries.extend(entries)

    # Filter by threshold and sort
    long = [e for e in all_entries if e["line_count"] >= threshold]
    long.sort(key=lambda e: -e["line_count"])

    return long, files_scanned

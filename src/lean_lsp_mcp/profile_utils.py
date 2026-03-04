"""Lean proof profiling via CLI trace output."""

import asyncio
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from lean_lsp_mcp.models import LineProfile, ProofProfileResult

_TRACE_RE = re.compile(r"^(\s*)\[([^\]]+)\]\s+\[([\d.]+)\]\s+(.+)$")
_CUMULATIVE_RE = re.compile(r"^\s+(\S+(?:\s+\S+)*)\s+([\d.]+)(ms|s)$")
_DECL_RE = re.compile(r"^\s*(?:private\s+)?(theorem|lemma|def)\s+(\S+)")
_HEADER_RE = re.compile(r"^(import|open|set_option|universe|variable)\s")
_SKIP_CATEGORIES = {"import", "initialization", "parsing", "interpretation", "linting"}


def _find_header_end(lines: list[str]) -> int:
    """Find where imports/header ends and declarations begin."""
    header_end, in_block = 0, False
    for i, line in enumerate(lines):
        s = line.strip()
        if "/-" in line:
            in_block = True
        if "-/" in line:
            in_block = False
        if in_block or not s or s.startswith("--") or _HEADER_RE.match(line):
            header_end = i + 1
        elif s.startswith(("namespace", "section")):
            header_end = i + 1
        elif _DECL_RE.match(line) or s.startswith(("@[", "private ", "protected ")):
            break
        else:
            header_end = i + 1
    return header_end


def _find_theorem_end(lines: list[str], start: int) -> int:
    """Find where theorem ends (next declaration or EOF)."""
    for i in range(start + 1, len(lines)):
        if _DECL_RE.match(lines[i]):
            return i
    return len(lines)


def _extract_theorem_source(lines: list[str], target_line: int) -> tuple[str, str, int]:
    """Extract imports/header + single theorem. Returns (source, name, theorem_start_in_source)."""
    m = _DECL_RE.match(lines[target_line - 1])
    if not m:
        raise ValueError(f"No theorem/lemma/def at line {target_line}")

    header_end = _find_header_end(lines)
    theorem_end = _find_theorem_end(lines, target_line - 1)

    header = "\n".join(lines[:header_end])
    theorem = "\n".join(lines[target_line - 1 : theorem_end])
    return f"{header}\n\n{theorem}\n", m.group(2), header_end + 2


def _parse_output(
    output: str,
) -> tuple[list[tuple[int, str, float, str]], dict[str, float]]:
    """Parse trace output into (traces, cumulative). Traces are (depth, cls, ms, msg)."""
    traces, cumulative, in_cumulative = [], {}, False

    for line in output.splitlines():
        if "cumulative profiling times:" in line:
            in_cumulative = True
        elif in_cumulative and (m := _CUMULATIVE_RE.match(line)):
            cat, val, unit = m.groups()
            cumulative[cat] = float(val) * (1000 if unit == "s" else 1)
        elif not in_cumulative and (m := _TRACE_RE.match(line)):
            indent, cls, time_s, msg = m.groups()
            traces.append((len(indent) // 2, cls, float(time_s) * 1000, msg))

    return traces, cumulative


def _build_proof_items(
    source_lines: list[str], proof_start: int
) -> list[tuple[int, str, bool]]:
    """Build list of (line_no, content, is_bullet) for proof lines."""
    items = []
    for i in range(proof_start, len(source_lines)):
        s = source_lines[i].strip()
        if s and not s.startswith("--"):
            is_bullet = s[0] in "·*-"
            items.append((i + 1, s.lstrip("·*- \t"), is_bullet))
    return items


def _match_line(
    tactic: str, is_bullet: bool, items: list[tuple[int, str, bool]], used: set[int]
) -> int | None:
    """Find matching source line for a tactic trace. Returns line number or None."""
    for ln, content, src_bullet in items:
        if ln in used:
            continue
        if is_bullet and src_bullet:
            return ln
        if (
            not is_bullet
            and content
            and (tactic.startswith(content[:25]) or content.startswith(tactic[:25]))
        ):
            return ln
    return None


def _extract_line_times(
    traces: list[tuple[int, str, float, str]],
    name: str,
    proof_items: list[tuple[int, str, bool]],
) -> tuple[dict[int, float], float]:
    """Extract per-line timing from traces."""
    line_times: dict[int, float] = defaultdict(float)
    total, value_depth, in_value, tactic_depth = 0.0, 0, False, None
    name_re = re.compile(rf"\b{re.escape(name)}\b")
    used: set[int] = set()

    for depth, cls, ms, msg in traces:
        if cls == "Elab.definition.value" and name_re.search(msg):
            in_value, value_depth, total = True, depth, ms
        elif cls == "Elab.async" and f"proof of {name}" in msg:
            total = max(total, ms)
        elif in_value:
            if depth <= value_depth:
                break
            if cls == "Elab.step" and not msg.startswith("expected type:"):
                tactic_depth = tactic_depth or depth
                if depth == tactic_depth:
                    tactic = msg.split("\n")[0].strip().lstrip("·*- \t")
                    if ln := _match_line(tactic, not tactic, proof_items, used):
                        line_times[ln] += ms
                        used.add(ln)

    return dict(line_times), total


def _filter_categories(cumulative: dict[str, float]) -> dict[str, float]:
    """Filter to relevant categories >= 1ms."""
    return {
        k: round(v, 1)
        for k, v in sorted(cumulative.items(), key=lambda x: -x[1])
        if k not in _SKIP_CATEGORIES and v >= 1.0
    }


async def _run_lean_profile(file_path: Path, project_path: Path, timeout: float) -> str:
    """Run lean --profile, return output."""
    proc = await asyncio.create_subprocess_exec(
        "lake",
        "env",
        "lean",
        "--profile",
        "-Dtrace.profiler=true",
        "-Dtrace.profiler.threshold=0",
        str(file_path.resolve()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=project_path.resolve(),
        env=os.environ.copy(),
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout.decode("utf-8", errors="replace")
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Profiling timed out after {timeout}s")


def _find_proof_start(source_lines: list[str]) -> int:
    """Find line after ':= by' in source."""
    for i, line in enumerate(source_lines):
        if ":= by" in line or line.rstrip().endswith(" by"):
            return i + 1
    raise ValueError("No 'by' proof found in theorem")


async def profile_theorem(
    file_path: Path,
    theorem_line: int,
    project_path: Path,
    timeout: float = 60.0,
    top_n: int = 5,
) -> ProofProfileResult:
    """Profile a theorem via `lean --profile`. Returns per-line timing data."""
    lines = file_path.read_text(encoding="utf-8").splitlines()
    if not (0 < theorem_line <= len(lines)):
        raise ValueError(f"Line {theorem_line} out of range")

    source, name, src_start = _extract_theorem_source(lines, theorem_line)
    source_lines = source.splitlines()
    line_offset = theorem_line - src_start
    proof_start = _find_proof_start(source_lines)
    proof_items = _build_proof_items(source_lines, proof_start)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", dir=project_path, delete=False, encoding="utf-8"
    ) as f:
        f.write(source)
        temp_path = Path(f.name)

    try:
        output = await _run_lean_profile(temp_path, project_path, timeout)
    finally:
        temp_path.unlink(missing_ok=True)

    traces, cumulative = _parse_output(output)
    line_times, total = _extract_line_times(traces, name, proof_items)

    top_lines = sorted(
        [(ln, ms) for ln, ms in line_times.items() if ms >= total * 0.01],
        key=lambda x: -x[1],
    )[:top_n]

    return ProofProfileResult(
        ms=round(total, 1),
        lines=[
            LineProfile(
                line=ln + line_offset,
                ms=round(ms, 1),
                text=source_lines[ln - 1].strip()[:60]
                if ln <= len(source_lines)
                else "",
            )
            for ln, ms in top_lines
        ],
        categories=_filter_categories(cumulative),
    )

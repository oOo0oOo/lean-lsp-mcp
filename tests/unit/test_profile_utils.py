"""Unit tests for profile_utils module."""

import os
import platform
from pathlib import Path

import pytest

from lean_lsp_mcp import profile_utils
from lean_lsp_mcp.profile_utils import (
    _build_proof_items,
    _extract_line_times,
    _extract_theorem_source,
    _filter_categories,
    _parse_output,
    _run_lean_profile,
)


class TestParseOutput:
    def test_parses_single_trace(self):
        output = "[Elab.command] [0.005] some message\n"
        traces, _ = _parse_output(output)
        assert traces == [(0, "Elab.command", 5.0, "some message")]

    def test_parses_indented_traces(self):
        output = "[A] [0.001] outer\n  [B] [0.002] inner\n"
        traces, _ = _parse_output(output)
        assert [(d, c) for d, c, _, _ in traces] == [(0, "A"), (1, "B")]

    def test_parses_cumulative_ms_and_s(self):
        output = "cumulative profiling times:\n        simp 2ms\n        import 4.5s\n"
        _, cumulative = _parse_output(output)
        assert cumulative == {"simp": 2.0, "import": 4500.0}

    def test_ignores_non_trace_lines(self):
        output = "random text\n[A] [0.001] msg\nmore text\n"
        traces, _ = _parse_output(output)
        assert len(traces) == 1


class TestBuildProofItems:
    def test_builds_items_from_proof_lines(self):
        source = ["theorem foo := by", "  simp", "  · ring", ""]
        items = _build_proof_items(source, 1)
        # (line_no, content, is_bullet)
        assert items == [(2, "simp", False), (3, "ring", True)]

    def test_skips_comments(self):
        source = ["by", "  -- comment", "  simp"]
        items = _build_proof_items(source, 1)
        assert items == [(3, "simp", False)]


class TestExtractLineTimes:
    def test_extracts_time_for_matching_line(self):
        traces = [
            (0, "Elab.definition.value", 10.0, "my_thm"),
            (1, "Elab.step", 8.0, "simp [a, b]"),
        ]
        proof_items = [(2, "simp [a, b]", False)]
        line_times, total = _extract_line_times(traces, "my_thm", proof_items)
        assert total == 10.0
        assert line_times == {2: 8.0}

    def test_matches_bullets(self):
        traces = [
            (0, "Elab.definition.value", 10.0, "my_thm"),
            (1, "Elab.step", 5.0, "·"),  # bullet trace (empty after strip)
        ]
        proof_items = [(3, "ring", True)]  # is_bullet=True
        line_times, _ = _extract_line_times(traces, "my_thm", proof_items)
        assert line_times == {3: 5.0}

    def test_filters_expected_type(self):
        traces = [
            (0, "Elab.definition.value", 10.0, "my_thm"),
            (1, "Elab.step", 5.0, "expected type: Prop"),
            (1, "Elab.step", 8.0, "ring"),
        ]
        proof_items = [(2, "ring", False)]
        line_times, _ = _extract_line_times(traces, "my_thm", proof_items)
        assert line_times == {2: 8.0}

    def test_no_match_returns_empty(self):
        traces = [(0, "Elab.definition.value", 10.0, "other")]
        line_times, total = _extract_line_times(traces, "missing", [])
        assert line_times == {}
        assert total == 0.0


class TestExtractTheoremSource:
    def test_extracts_single_theorem_with_imports(self):
        lines = [
            "import Mathlib",
            "open Nat",
            "",
            "theorem foo : True := by trivial",
            "",
            "theorem bar : True := by trivial",
        ]
        source, name, theorem_start = _extract_theorem_source(lines, 4)
        assert name == "foo"
        assert "import Mathlib" in source and "open Nat" in source
        assert "theorem foo" in source and "theorem bar" not in source
        # theorem is at line 4 in original, source has: 1:import 2:open 3:blank 4:blank 5:theorem
        assert theorem_start == 5

    def test_raises_on_non_theorem(self):
        with pytest.raises(ValueError, match="No theorem"):
            _extract_theorem_source(["-- comment"], 1)


class TestFilterCategories:
    def test_filters_skip_and_small(self):
        cumulative = {"import": 100.0, "simp": 5.0, "parsing": 2.0, "ring": 0.5}
        assert _filter_categories(cumulative) == {"simp": 5.0}


class _FakeProc:
    """Stands in for the `lake env lean --profile` subprocess."""

    pid = 4321

    def __init__(self):
        self.killed = False

    async def communicate(self):
        return b"[Elab.command] [0.005] done\n", None

    def kill(self):
        self.killed = True


class TestRunLeanProfileCleanup:
    """The cleanup in _run_lean_profile's `finally` runs on every profile.

    Both platforms are simulated rather than detected, so each case is
    exercised wherever the suite runs.
    """

    @pytest.fixture
    def proc(self, monkeypatch) -> _FakeProc:
        fake = _FakeProc()

        async def fake_exec(*args, **kwargs):
            return fake

        monkeypatch.setattr(profile_utils.asyncio, "create_subprocess_exec", fake_exec)
        return fake

    async def test_windows_kills_the_process_instead_of_the_group(
        self, monkeypatch, tmp_path: Path, proc: _FakeProc
    ):
        """Windows has no os.killpg, so the group kill must not be attempted.

        Calling it raised AttributeError out of the `finally`, which discarded
        the profile that had already been produced.
        """
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.delattr(os, "killpg", raising=False)
        monkeypatch.delattr(os, "getpgid", raising=False)

        output = await _run_lean_profile(tmp_path / "T.lean", tmp_path, 30.0)

        assert output == "[Elab.command] [0.005] done\n"
        assert proc.killed

    async def test_posix_still_kills_the_whole_group(
        self, monkeypatch, tmp_path: Path, proc: _FakeProc
    ):
        """The group kill is what stops lean surviving as an orphan."""
        killed: list[tuple[int, int]] = []
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(os, "getpgid", lambda pid: pid, raising=False)
        monkeypatch.setattr(
            os, "killpg", lambda pgid, sig: killed.append((pgid, sig)), raising=False
        )

        output = await _run_lean_profile(tmp_path / "T.lean", tmp_path, 30.0)

        assert output == "[Elab.command] [0.005] done\n"
        assert killed == [(_FakeProc.pid, 9)]
        assert not proc.killed

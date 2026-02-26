"""Unit tests for verify module."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lean_lsp_mcp.verify import (
    check_axiom_errors,
    parse_axioms,
    scan_warnings,
)

_has_rg = shutil.which("rg") is not None


class TestParseAxioms:
    def test_single_axiom(self):
        diags = [{"severity": 3, "message": "'x' depends on axioms: [propext]"}]
        assert parse_axioms(diags) == ["propext"]

    def test_multiple_axioms(self):
        diags = [
            {
                "severity": 3,
                "message": "'x' depends on axioms: [propext, Classical.choice]",
            }
        ]
        assert parse_axioms(diags) == ["propext", "Classical.choice"]

    def test_sorry(self):
        diags = [{"severity": 3, "message": "'x' depends on axioms: [sorryAx]"}]
        assert parse_axioms(diags) == ["sorryAx"]

    def test_no_axioms(self):
        diags = [{"severity": 3, "message": "'x' does not depend on any axioms"}]
        assert parse_axioms(diags) == []

    def test_ignores_non_info(self):
        diags = [{"severity": 1, "message": "'x' depends on axioms: [sorryAx]"}]
        assert parse_axioms(diags) == []

    def test_multiline_axioms(self):
        """Long declaration names cause #print axioms to wrap across lines."""
        diags = [
            {
                "severity": 3,
                "message": (
                    "'Foo.integralTensorPower_coherentState_eq'"
                    " depends on axioms: [propext,\n"
                    " sorryAx,\n"
                    " Classical.choice,\n"
                    " Quot.sound]"
                ),
            }
        ]
        assert parse_axioms(diags) == [
            "propext",
            "sorryAx",
            "Classical.choice",
            "Quot.sound",
        ]

    def test_empty(self):
        assert parse_axioms([]) == []


class TestCheckAxiomErrors:
    def test_returns_none_on_success(self):
        assert check_axiom_errors([{"severity": 3, "message": "info"}]) is None

    def test_returns_errors(self):
        diags = [{"severity": 1, "message": "unknown id 'foo'"}]
        assert check_axiom_errors(diags) == "unknown id 'foo'"

    def test_joins_multiple(self):
        diags = [
            {"severity": 1, "message": "a"},
            {"severity": 1, "message": "b"},
        ]
        assert check_axiom_errors(diags) == "a; b"


class TestScanWarnings:
    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_finds_patterns(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text(
            "set_option debug.skipKernelTC true\nunsafe def foo := 1\ntheorem bar : True := trivial\n"
        )
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("debug." in p for p in patterns)
        assert any("unsafe" in p for p in patterns)

    def test_clean_file(self, tmp_path: Path):
        f = tmp_path / "clean.lean"
        f.write_text("theorem clean : True := trivial\n")
        assert scan_warnings(f) == []

    def test_nonexistent(self, tmp_path: Path):
        assert scan_warnings(tmp_path / "nope.lean") == []

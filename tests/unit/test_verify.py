"""Unit tests for verify module."""

from __future__ import annotations

from pathlib import Path

from lean_lsp_mcp.verify import (
    check_axiom_errors,
    make_axiom_check,
    parse_axioms,
    scan_warnings,
)


class TestParseAxioms:
    def test_standard(self):
        diags = [
            {
                "severity": 3,
                "message": "'x' depends on axioms:\n  [propext]\n  [Classical.choice]",
            }
        ]
        assert parse_axioms(diags) == ["propext", "Classical.choice"]

    def test_sorry(self):
        diags = [{"severity": 3, "message": "'x' depends on axioms:\n  [sorryAx]"}]
        assert parse_axioms(diags) == ["sorryAx"]

    def test_no_axioms(self):
        diags = [{"severity": 3, "message": "'x' does not depend on any axioms"}]
        assert parse_axioms(diags) == []

    def test_ignores_non_info(self):
        diags = [{"severity": 1, "message": "'x' depends on axioms:\n  [sorryAx]"}]
        assert parse_axioms(diags) == []

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


class TestMakeAxiomCheck:
    def test_creates_temp_file(self, tmp_path: Path):
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "lean-toolchain").touch()
        src = proj / "Foo.lean"
        src.write_text("theorem bar : True := trivial\n")
        rel_path, tmp = make_axiom_check(src, proj, "bar")
        assert tmp.exists()
        content = tmp.read_text()
        assert "import Foo" in content
        assert "#print axioms bar" in content
        tmp.unlink()

    def test_raises_on_outside_project(self, tmp_path: Path):
        proj = tmp_path / "proj"
        proj.mkdir()
        outside = tmp_path / "other" / "Foo.lean"
        outside.parent.mkdir()
        outside.touch()
        try:
            make_axiom_check(outside, proj, "bar")
            assert False, "Should have raised"
        except ValueError:
            pass


class TestScanWarnings:
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

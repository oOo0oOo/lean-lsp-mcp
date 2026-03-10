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

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_implemented_by(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text("@[implemented_by nativeImpl] def foo := 1\n")
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("implemented_by" in p for p in patterns)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_extern(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text('@[extern "lean_io_ref"] def foo : IO Unit := sorry\n')
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("extern" in p for p in patterns)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_opaque(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text("opaque foo : Nat\n")
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("opaque" in p for p in patterns)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_local_instance(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text("local instance : Inhabited Nat := ⟨0⟩\n")
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("local instance" in p for p in patterns)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_local_notation(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text('local notation "x" => Nat\n')
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("local notation" in p for p in patterns)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_scoped_notation(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text('scoped notation "x" => Nat\n')
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("scoped notation" in p for p in patterns)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_scoped_instance(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text("scoped instance : Inhabited Nat := ⟨0⟩\n")
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("scoped instance" in p for p in patterns)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_csimp(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text("@[csimp] theorem foo : True := trivial\n")
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("csimp" in p for p in patterns)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_import_lean_elab(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text("import Lean.Elab.Command\n")
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("Lean.Elab" in p for p in patterns)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_import_lean_meta(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text("import Lean.Meta.Basic\n")
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("Lean.Meta" in p for p in patterns)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_local_macro_rules(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text("local macro_rules | `(tactic| trivial) => `(tactic| rfl)\n")
        patterns = [w["pattern"] for w in scan_warnings(f)]
        assert any("local macro_rules" in p for p in patterns)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_multiple_patterns(self, tmp_path: Path):
        f = tmp_path / "test.lean"
        f.write_text(
            "unsafe def foo := 1\n"
            '@[extern "bar"] opaque baz : Nat\n'
            "set_option debug.skipKernelTC true\n"
        )
        warnings = scan_warnings(f)
        assert len(warnings) >= 3

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_pattern_in_comment_false_positive(self, tmp_path: Path):
        """rg doesn't know about comments — this is a known limitation."""
        f = tmp_path / "test.lean"
        f.write_text("-- unsafe def foo := 1\ntheorem bar : True := trivial\n")
        warnings = scan_warnings(f)
        # rg matches inside comments — this is a false positive
        assert any("unsafe" in w["pattern"] for w in warnings)

    @pytest.mark.skipif(not _has_rg, reason="ripgrep not installed")
    def test_pattern_in_string_false_positive(self, tmp_path: Path):
        """rg doesn't know about strings — this is a known limitation."""
        f = tmp_path / "test.lean"
        f.write_text('def msg := "unsafe code here"\n')
        warnings = scan_warnings(f)
        assert any("unsafe" in w["pattern"] for w in warnings)

    def test_rg_timeout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """TimeoutExpired during rg execution returns empty list."""
        import subprocess as sp

        def fake_run(*args, **kwargs):
            raise sp.TimeoutExpired(cmd="rg", timeout=10)

        monkeypatch.setattr(sp, "run", fake_run)
        f = tmp_path / "test.lean"
        f.write_text("unsafe def foo := 1\n")
        assert scan_warnings(f) == []

    def test_rg_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """FileNotFoundError (rg not installed) returns empty list."""
        import subprocess as sp

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("rg not found")

        monkeypatch.setattr(sp, "run", fake_run)
        f = tmp_path / "test.lean"
        f.write_text("unsafe def foo := 1\n")
        assert scan_warnings(f) == []


class TestParseAxiomsEdgeCases:
    def test_sorry_ax_in_axioms(self):
        """sorryAx appears in axiom list — caller must check for it."""
        diags = [
            {"severity": 3, "message": "'f' depends on axioms: [sorryAx, propext]"}
        ]
        axioms = parse_axioms(diags)
        assert "sorryAx" in axioms
        assert "propext" in axioms

    def test_no_axioms_declaration(self):
        """Message saying no axiom dependency."""
        diags = [{"severity": 3, "message": "'f' does not depend on any axioms"}]
        assert parse_axioms(diags) == []

    def test_empty_axiom_list(self):
        """Empty brackets — regex doesn't match (.+? requires at least one char)."""
        diags = [{"severity": 3, "message": "'f' depends on axioms: []"}]
        assert parse_axioms(diags) == []

    def test_missing_severity(self):
        """Diagnostic without severity — should not crash."""
        diags = [{"message": "'f' depends on axioms: [propext]"}]
        assert parse_axioms(diags) == []

    def test_missing_message(self):
        """Diagnostic without message — should not crash."""
        diags = [{"severity": 3}]
        assert parse_axioms(diags) == []

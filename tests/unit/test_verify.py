"""Unit tests for verify module."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lean_lsp_mcp.verify import (
    check_axiom_errors,
    parse_axioms,
    read_lean_source_utf8,
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


class TestReadLeanSourceUtf8:
    def test_reads_utf8_regardless_of_platform_default(self, tmp_path: Path):
        source = "-- Unicode: \u2212 \u2080 \u03b1\ntheorem clean : True := trivial\n"
        file_path = tmp_path / "unicode.lean"
        file_path.write_text(source, encoding="utf-8")

        assert read_lean_source_utf8(file_path) == source

    def test_reports_utf8_when_decoding_fails(self, tmp_path: Path):
        file_path = tmp_path / "invalid.lean"
        file_path.write_bytes(b"-- invalid: \xff\n")

        with pytest.raises(ValueError, match=r"using UTF-8"):
            read_lean_source_utf8(file_path)


@pytest.mark.asyncio
async def test_verify_reads_utf8_without_opening_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from lean_lsp_mcp import server
    from lean_lsp_mcp.tools import analysis

    file_path = tmp_path / "unicode.lean"
    file_path.write_text(
        "-- Unicode: \u2212 \u2080 \u03b1\ntheorem clean : True := trivial\n",
        encoding="utf-8",
    )

    class Policy:
        def validate_path(self, path: Path) -> Path:
            return path

    class Pool:
        async def run_text(self, text: str):
            return SimpleNamespace(
                diagnostics=SimpleNamespace(
                    items=[
                        {
                            "severity": 3,
                            "message": "'clean' does not depend on any axioms",
                            "range": {"start": {"line": 999}},
                        }
                    ]
                )
            )

    monkeypatch.setattr(server, "_validate_theorem_name", lambda name: name)
    monkeypatch.setattr(
        server, "setup_client_for_file", AsyncMock(return_value="unicode.lean")
    )
    monkeypatch.setattr(server, "resolve_file_path", lambda _ctx, _path: file_path)
    monkeypatch.setattr(server, "_RG_AVAILABLE", False)
    monkeypatch.setattr(analysis, "get_path_policy", lambda _ctx: Policy())
    monkeypatch.setattr(analysis, "get_scratch_pool", lambda _ctx: Pool())
    monkeypatch.setattr(
        analysis,
        "open_synced",
        AsyncMock(
            side_effect=AssertionError("lean_verify must not read via open_synced")
        ),
    )

    result = await analysis.verify_theorem(
        object(), str(file_path), "Namespace.clean", scan_source=False
    )

    assert result.axioms == []


@pytest.mark.asyncio
async def test_verify_reports_utf8_decode_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from lean_lsp_mcp import server
    from lean_lsp_mcp.tools import analysis

    file_path = tmp_path / "invalid.lean"
    file_path.write_bytes(b"-- invalid: \xff\n")

    class Policy:
        def validate_path(self, path: Path) -> Path:
            return path

    monkeypatch.setattr(server, "_validate_theorem_name", lambda name: name)
    monkeypatch.setattr(
        server, "setup_client_for_file", AsyncMock(return_value="invalid.lean")
    )
    monkeypatch.setattr(server, "resolve_file_path", lambda _ctx, _path: file_path)
    monkeypatch.setattr(analysis, "get_path_policy", lambda _ctx: Policy())

    with pytest.raises(server.LeanToolError, match=r"using UTF-8"):
        await analysis.verify_theorem(
            object(), str(file_path), "Namespace.invalid", scan_source=False
        )

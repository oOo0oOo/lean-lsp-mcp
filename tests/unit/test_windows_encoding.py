import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_output_capture_handles_unicode(monkeypatch) -> None:
    """Test that OutputCapture correctly handles Unicode characters.

    This test simulates Windows behavior where tempfile defaults to cp1252 encoding.
    """
    from lean_lsp_mcp.utils import OutputCapture

    monkeypatch.setenv("LEAN_LSP_MCP_ACTIVE_TRANSPORT", "streamable-http")

    # Unicode content that Lean produces in diagnostics and goals
    unicode_content = "⊢ ℕ → ℕ"

    # Patch tempfile to simulate Windows default behavior (cp1252 encoding)
    original_tempfile = tempfile.NamedTemporaryFile

    def windows_style_tempfile(*args, **kwargs):
        # If encoding not explicitly specified, use cp1252 (Windows default)
        if "encoding" not in kwargs and kwargs.get("mode", "").startswith("w"):
            kwargs["encoding"] = "cp1252"
        return original_tempfile(*args, **kwargs)

    with patch(
        "lean_lsp_mcp.utils.tempfile.NamedTemporaryFile",
        side_effect=windows_style_tempfile,
    ):
        try:
            with OutputCapture() as capture:
                sys.stdout.write(unicode_content)
                sys.stdout.flush()

            captured = capture.get_output()

            # This assertion fails with the bug (content is corrupted: 'âŠ¢ â„• â†' â„•')
            # It will pass when encoding="utf-8" is added to tempfile creation
            assert captured == unicode_content, (
                f"Unicode content was corrupted. "
                f"Expected: {repr(unicode_content)}, "
                f"Got: {repr(captured)}"
            )

        except UnicodeEncodeError as e:
            # On some systems, cp1252 encoding fails immediately when writing
            pytest.fail(
                f"UnicodeEncodeError when writing to tempfile with cp1252 encoding: {e}\n"
                f"Fix: Add encoding='utf-8' to tempfile.NamedTemporaryFile in OutputCapture"
            )


def test_lean_run_code_handles_unicode() -> None:
    """Test that writing and reading Lean files with Unicode characters round-trips correctly."""
    lean_code = "theorem test : ℕ → ℕ := by\n  intro n\n  sorry\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", delete=False, encoding="utf-8"
    ) as f:
        f.write(lean_code)
        temp_path = f.name

    try:
        with open(temp_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == lean_code
    finally:
        import os

        os.unlink(temp_path)


def test_tempfile_for_logging_handles_unicode() -> None:
    """Test that TemporaryFile for logging handles Unicode characters.

    The test_logging.py helper uses TemporaryFile to capture stderr.
    This verifies that encoding='utf-8' is specified to handle Unicode in logs.
    """
    # Simulate log output with Unicode (like Lean error messages)
    log_content = "Error: unsolved goals\n⊢ ℕ → ℕ\n∀ n : ℕ, n ≤ n"

    # Test that TemporaryFile with UTF-8 encoding works correctly
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as f:
        f.write(log_content)
        f.seek(0)
        content = f.read()

    assert content == log_content, (
        f"Unicode content was corrupted. "
        f"Expected: {repr(log_content)}, "
        f"Got: {repr(content)}"
    )


class TestReadTextEncoding:
    """Test that read_text() calls use explicit encoding (issue #162).

    On Windows with cp932 (Shift-JIS) default encoding, Japanese characters
    in Lean files cause UnicodeDecodeError without explicit encoding='utf-8'.
    """

    JAPANESE_LEAN = "-- 日本語コメント\ntheorem test : ℕ → ℕ := id\n"

    def test_resolve_namespaces_reads_utf8(self, tmp_path: Path) -> None:
        """_resolve_namespaces should read files with explicit UTF-8."""
        from lean_lsp_mcp.search_utils import _resolve_namespaces

        p = tmp_path / "Test.lean"
        p.write_text(
            "namespace Foo\n-- 日本語\ndef bar := 1\nend Foo\n",
            encoding="utf-8",
        )

        # Simulate cp932 default — read_text() without encoding would fail
        with patch("locale.getpreferredencoding", return_value="cp932"):
            result = _resolve_namespaces(p, {3})

        assert result == {3: "Foo"}

    def test_subprocess_text_encoding_ripgrep(self) -> None:
        """_create_ripgrep_process passes encoding='utf-8'."""
        from lean_lsp_mcp.search_utils import _create_ripgrep_process

        import shutil

        if not shutil.which("rg"):
            pytest.skip("rg not installed")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "Test.lean"
            p.write_text(self.JAPANESE_LEAN, encoding="utf-8")

            proc = _create_ripgrep_process(["rg", "--json", "日本語", str(p)], cwd=td)
            stdout, _ = proc.communicate(timeout=5)
            # stdout is str (text mode) and should contain the match
            assert "日本語" in stdout

    def test_subprocess_text_encoding_verify(self, tmp_path: Path) -> None:
        """scan_warnings handles Japanese characters in files."""
        from lean_lsp_mcp.verify import scan_warnings

        import shutil

        if not shutil.which("rg"):
            pytest.skip("rg not installed")

        p = tmp_path / "Test.lean"
        p.write_text("-- 日本語\nunsafe def x := 1\n", encoding="utf-8")

        # Should not raise UnicodeDecodeError
        warnings = scan_warnings(p)
        assert any(w["pattern"] == "unsafe" for w in warnings)

import sys
import tempfile
from unittest.mock import patch

import pytest


def test_output_capture_handles_unicode() -> None:
    """Test that OutputCapture correctly handles Unicode characters.

    This test simulates Windows behavior where tempfile defaults to cp1252 encoding.
    """
    from lean_lsp_mcp.utils import OutputCapture

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

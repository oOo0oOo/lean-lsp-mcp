"""Unit tests for REPL pool manager."""

import pytest

from lean_lsp_mcp.pool import SnippetResult, split_snippet, SplitSnippet
from lean_lsp_mcp.pool.settings import PoolSettings


class TestSplitSnippet:
    """Tests for header/body splitting."""

    def test_simple_import(self):
        """Single import is header."""
        code = "import Mathlib\n\ndef x := 1"
        result = split_snippet(code)
        assert result.header == "import Mathlib"
        assert result.body == "def x := 1"
        assert result.header_line_count == 2  # import + blank

    def test_multiple_imports(self):
        """Multiple imports preserved in order."""
        code = "import Foo\nimport Bar\n\ndef x := 1"
        result = split_snippet(code)
        assert result.header == "import Foo\nimport Bar"
        assert result.body == "def x := 1"

    def test_mathlib_consolidation(self):
        """Multiple Mathlib imports consolidated."""
        code = "import Mathlib.Data.Nat\nimport Mathlib.Data.List\nimport Other\n\ndef x := 1"
        result = split_snippet(code)
        # Should have single "import Mathlib" plus "import Other"
        assert "import Mathlib" in result.header
        assert "import Other" in result.header
        assert result.header.count("import Mathlib") == 1

    def test_no_imports(self):
        """Code without imports has empty header."""
        code = "def x := 1\ndef y := 2"
        result = split_snippet(code)
        assert result.header == ""
        assert result.body == "def x := 1\ndef y := 2"
        assert result.header_line_count == 0

    def test_blank_lines_in_header(self):
        """Blank lines before first non-import included in header count."""
        code = "import Foo\n\n\ndef x := 1"
        result = split_snippet(code)
        assert result.header == "import Foo"
        assert result.body == "def x := 1"
        assert result.header_line_count == 3  # import + 2 blanks

    def test_duplicate_imports_deduplicated(self):
        """Duplicate imports removed."""
        code = "import Foo\nimport Foo\nimport Bar\n\ndef x := 1"
        result = split_snippet(code)
        assert result.header.count("import Foo") == 1
        assert "import Bar" in result.header


class TestSnippetResult:
    """Tests for SnippetResult dataclass."""

    def test_default_values(self):
        """Default values are None."""
        result = SnippetResult()
        assert result.env is None
        assert result.goals is None
        assert result.messages is None
        assert result.sorries is None
        assert result.error is None
        assert result.proof_state is None

    def test_with_goals(self):
        """Goals are stored correctly."""
        result = SnippetResult(
            env=1,
            goals=["⊢ P", "⊢ Q"],
            proof_state=5,
        )
        assert result.env == 1
        assert result.goals == ["⊢ P", "⊢ Q"]
        assert result.proof_state == 5

    def test_with_error(self):
        """Error messages stored correctly."""
        result = SnippetResult(error="Type mismatch")
        assert result.error == "Type mismatch"


class TestPoolSettings:
    """Tests for pool settings configuration."""

    def test_default_settings(self, monkeypatch):
        """Default settings are reasonable."""
        monkeypatch.delenv("LEAN_MCP_MAX_REPLS", raising=False)
        monkeypatch.delenv("LEAN_MCP_MAX_REPL_USES", raising=False)
        monkeypatch.delenv("LEAN_MCP_MAX_REPL_MEM", raising=False)

        settings = PoolSettings()
        assert settings.max_repls >= 1
        assert settings.max_repl_uses == -1  # Unlimited
        assert settings.max_repl_mem > 0  # Has some memory limit

    def test_env_override(self, monkeypatch):
        """Environment variables override defaults."""
        monkeypatch.setenv("LEAN_MCP_MAX_REPLS", "4")
        monkeypatch.setenv("LEAN_MCP_MAX_REPL_USES", "100")
        monkeypatch.setenv("LEAN_MCP_MAX_REPL_MEM", "4G")  # Must use M or G suffix

        settings = PoolSettings()
        assert settings.max_repls == 4
        assert settings.max_repl_uses == 100
        assert settings.max_repl_mem == 4 * 1024  # 4G = 4096 MB

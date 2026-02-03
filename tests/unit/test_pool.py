"""Unit tests for REPL pool."""

from lean_lsp_mcp.pool import SnippetResult, split_code
from lean_lsp_mcp.pool.settings import PoolSettings


class TestSplitCode:
    def test_simple_import(self):
        code = "import Mathlib\n\ndef x := 1"
        result = split_code(code)
        assert result.header == "import Mathlib"
        assert result.body == "def x := 1"

    def test_multiple_imports(self):
        code = "import Foo\nimport Bar\n\ndef x := 1"
        result = split_code(code)
        assert result.header == "import Foo\nimport Bar"
        assert result.body == "def x := 1"

    def test_mathlib_consolidation(self):
        code = "import Mathlib.Data.Nat\nimport Mathlib.Data.List\nimport Other\n\ndef x := 1"
        result = split_code(code)
        assert "import Mathlib" in result.header
        assert "import Other" in result.header
        assert result.header.count("import Mathlib") == 1

    def test_no_imports(self):
        code = "def x := 1\ndef y := 2"
        result = split_code(code)
        assert result.header == ""
        assert result.body == "def x := 1\ndef y := 2"

    def test_duplicate_imports_deduplicated(self):
        code = "import Foo\nimport Foo\nimport Bar\n\ndef x := 1"
        result = split_code(code)
        assert result.header.count("import Foo") == 1
        assert "import Bar" in result.header


class TestSnippetResult:
    def test_default_values(self):
        result = SnippetResult()
        assert result.goals == []
        assert result.messages == []
        assert result.error is None

    def test_with_goals(self):
        result = SnippetResult(goals=["⊢ P", "⊢ Q"])
        assert result.goals == ["⊢ P", "⊢ Q"]

    def test_with_error(self):
        result = SnippetResult(error="Type mismatch")
        assert result.error == "Type mismatch"


class TestPoolSettings:
    def test_default_settings(self, monkeypatch):
        monkeypatch.delenv("LEAN_REPL_WORKERS", raising=False)
        monkeypatch.delenv("LEAN_REPL_TIMEOUT", raising=False)
        monkeypatch.delenv("LEAN_REPL_MEM", raising=False)
        monkeypatch.delenv("LEAN_REPL_PATH", raising=False)

        settings = PoolSettings.from_env()
        assert settings.workers >= 1
        assert settings.timeout == 60
        assert settings.mem_mb > 0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LEAN_REPL_WORKERS", "4")
        monkeypatch.setenv("LEAN_REPL_TIMEOUT", "120")
        monkeypatch.setenv("LEAN_REPL_MEM", "4G")

        settings = PoolSettings.from_env()
        assert settings.workers == 4
        assert settings.timeout == 120
        assert settings.mem_mb == 4 * 1024

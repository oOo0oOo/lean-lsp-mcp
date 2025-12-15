"""Tests for local leansearch functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lean_lsp_mcp.leansearch import (
    LeanSearchManager,
    _compute_project_hash,
    _extract_declarations_from_file,
    _find_lean_files,
    check_leansearch_available,
    get_cache_dir,
)


class TestGetCacheDir:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("LEAN_LEANSEARCH_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: Path("/home/user"))
        assert get_cache_dir() == Path("/home/user/.cache/lean-lsp-mcp/leansearch")

    def test_xdg(self, monkeypatch):
        monkeypatch.delenv("LEAN_LEANSEARCH_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
        assert get_cache_dir() == Path("/xdg/lean-lsp-mcp/leansearch")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LEAN_LEANSEARCH_CACHE_DIR", "/custom")
        assert get_cache_dir() == Path("/custom")


class TestCheckLeansearchAvailable:
    def test_chromadb_available(self, monkeypatch):
        # Mock find_spec to return a non-None spec
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            available, msg = check_leansearch_available()
            assert available
            assert msg == ""

    def test_chromadb_missing(self, monkeypatch):
        # Mock find_spec to return None (module not found)
        with patch("importlib.util.find_spec", return_value=None):
            available, msg = check_leansearch_available()
            assert not available
            assert "chromadb" in msg


class TestComputeProjectHash:
    def test_hash_with_manifest(self, tmp_path):
        project = tmp_path / "myproject"
        project.mkdir()
        manifest = project / "lake-manifest.json"
        manifest.write_text('{"packages": []}')

        hash1 = _compute_project_hash(project)
        assert len(hash1) == 16  # 16 hex chars

        # Same content -> same hash
        hash2 = _compute_project_hash(project)
        assert hash1 == hash2

        # Different content -> different hash
        manifest.write_text('{"packages": [{"name": "mathlib"}]}')
        hash3 = _compute_project_hash(project)
        assert hash3 != hash1

    def test_hash_with_lakefile(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        lakefile = project / "lakefile.lean"
        lakefile.write_text("import Lake")

        hash1 = _compute_project_hash(project)
        assert len(hash1) == 16


class TestExtractDeclarations:
    def test_extract_theorem(self, tmp_path):
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text(
            """
theorem foo : 1 + 1 = 2 := by rfl

lemma bar (n : Nat) : n + 0 = n := by simp
"""
        )

        decls = _extract_declarations_from_file(lean_file, "Test")
        names = [d.name for d in decls]

        assert "Test.foo" in names
        assert "Test.bar" in names

    def test_extract_def(self, tmp_path):
        lean_file = tmp_path / "Defs.lean"
        lean_file.write_text(
            """
def myFunc (x : Nat) : Nat := x + 1

abbrev MyAlias := Nat

class MyClass where
  value : Nat
"""
        )

        decls = _extract_declarations_from_file(lean_file, "Defs")
        kinds = {d.name: d.kind for d in decls}

        assert "Defs.myFunc" in kinds
        assert kinds["Defs.myFunc"] == "def"
        assert "Defs.MyAlias" in kinds
        assert kinds["Defs.MyAlias"] == "abbrev"
        assert "Defs.MyClass" in kinds
        assert kinds["Defs.MyClass"] == "class"

    def test_extract_with_docstring(self, tmp_path):
        lean_file = tmp_path / "Documented.lean"
        lean_file.write_text(
            """
/-- This is a documented theorem. -/
theorem documented : True := trivial
"""
        )

        decls = _extract_declarations_from_file(lean_file, "Documented")
        assert len(decls) >= 1
        doc_decl = next((d for d in decls if "documented" in d.name.lower()), None)
        assert doc_decl is not None
        # Note: docstring extraction is best-effort with regex

    def test_skip_private(self, tmp_path):
        lean_file = tmp_path / "Private.lean"
        lean_file.write_text(
            """
def _privateHelper : Nat := 0

def publicFunc : Nat := _privateHelper
"""
        )

        decls = _extract_declarations_from_file(lean_file, "Private")
        names = [d.name for d in decls]

        # Private helpers starting with _ should be skipped
        assert not any("_privateHelper" in n for n in names)
        assert "Private.publicFunc" in names

    def test_empty_file(self, tmp_path):
        lean_file = tmp_path / "Empty.lean"
        lean_file.write_text("")

        decls = _extract_declarations_from_file(lean_file)
        assert decls == []

    def test_nonexistent_file(self, tmp_path):
        decls = _extract_declarations_from_file(tmp_path / "nonexistent.lean")
        assert decls == []


class TestFindLeanFiles:
    def test_find_files(self, tmp_path):
        # Create test structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "Main.lean").touch()
        (tmp_path / "src" / "Lib.lean").touch()
        (tmp_path / ".lake" / "build" / "ir").mkdir(parents=True)
        (tmp_path / ".lake" / "build" / "Test.lean").touch()  # Should be excluded

        files = _find_lean_files(tmp_path)
        names = [f.name for f in files]

        assert "Main.lean" in names
        assert "Lib.lean" in names
        assert "Test.lean" not in names  # Build dir excluded

    def test_exclude_build_by_default(self, tmp_path):
        (tmp_path / ".lake" / "build").mkdir(parents=True)
        (tmp_path / ".lake" / "build" / "Test.lean").touch()

        files = _find_lean_files(tmp_path, exclude_build=True)
        assert len(files) == 0

        files = _find_lean_files(tmp_path, exclude_build=False)
        assert len(files) == 1


class TestLeanSearchManager:
    @pytest.fixture
    def mgr(self, tmp_path):
        return LeanSearchManager(
            project_root=tmp_path / "project",
            cache_dir=tmp_path / "cache",
        )

    def test_init(self, mgr, tmp_path):
        assert mgr.project_root == tmp_path / "project"
        assert mgr.cache_dir == tmp_path / "cache"
        assert not mgr.is_ready

    def test_collection_name(self, mgr):
        name = mgr._get_collection_name()
        assert name.startswith("leansearch_project_")

    def test_collection_name_no_project(self, tmp_path):
        mgr = LeanSearchManager(project_root=None, cache_dir=tmp_path)
        assert mgr._get_collection_name() == "leansearch_global"

    def test_collect_files_empty_project(self, mgr):
        # No project dir exists
        files = mgr._collect_lean_files()
        assert files == []

    def test_collect_files_with_project(self, mgr):
        # Create project structure
        mgr.project_root.mkdir(parents=True)
        src = mgr.project_root / "src"
        src.mkdir()
        (src / "Main.lean").write_text("def main : IO Unit := pure ()")

        files = mgr._collect_lean_files()
        assert len(files) >= 1
        assert any("Main.lean" in str(f) for f, _, _ in files)

    def test_collect_files_with_lake_packages(self, mgr):
        # Create project with fake dependency
        mgr.project_root.mkdir(parents=True)
        pkg_dir = mgr.project_root / ".lake" / "packages" / "std"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "Std.lean").write_text("def stdFunc : Nat := 0")

        files = mgr._collect_lean_files()
        assert any("Std.lean" in str(f) for f, _, _ in files)


@pytest.mark.skipif(
    not check_leansearch_available()[0],
    reason="chromadb not installed",
)
class TestLeanSearchManagerWithChroma:
    """Tests that require chromadb to be installed."""

    @pytest.fixture
    def mgr(self, tmp_path):
        return LeanSearchManager(
            project_root=tmp_path / "project",
            cache_dir=tmp_path / "cache",
        )

    def test_index_empty_project(self, mgr):
        mgr.project_root.mkdir(parents=True)
        count = mgr.index_project()
        assert count == 0
        assert mgr.is_ready

    def test_index_and_search(self, mgr):
        # Create a minimal project with some declarations
        mgr.project_root.mkdir(parents=True)
        (mgr.project_root / "Test.lean").write_text(
            """
theorem add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b

def double (x : Nat) : Nat := x + x

lemma double_is_two_times (x : Nat) : double x = 2 * x := by omega
"""
        )

        # Index
        count = mgr.index_project()
        assert count >= 2  # At least add_comm and double

        # Search
        results = mgr.search("addition commutative", num_results=5)
        assert isinstance(results, list)
        # Results should include something related to addition

    def test_clear_cache(self, mgr):
        mgr.project_root.mkdir(parents=True)
        (mgr.project_root / "Test.lean").write_text("def foo : Nat := 0")
        mgr.index_project()
        assert mgr.is_ready

        mgr.clear_cache()
        assert not mgr.is_ready

    @pytest.mark.asyncio
    async def test_ensure_indexed(self, mgr):
        mgr.project_root.mkdir(parents=True)
        (mgr.project_root / "Test.lean").write_text("def bar : Nat := 1")

        result = await mgr.ensure_indexed()
        assert result
        assert mgr.is_ready

    @pytest.mark.asyncio
    async def test_ensure_indexed_project_change(self, mgr, tmp_path):
        # First project
        mgr.project_root.mkdir(parents=True)
        (mgr.project_root / "Test.lean").write_text("def first : Nat := 0")
        await mgr.ensure_indexed()
        assert mgr.is_ready

        # Change to new project
        new_project = tmp_path / "project2"
        new_project.mkdir()
        (new_project / "Test.lean").write_text("def second : Nat := 1")

        # Should re-index for new project
        await mgr.ensure_indexed(project_root=new_project)
        assert mgr.project_root == new_project


@pytest.mark.slow
class TestLeanSearchIntegration:
    """Integration tests with a real Lean project.

    Run with: pytest -m slow tests/unit/test_leansearch.py
    Requires chromadb and a test project in tests/test_project.
    """

    @pytest.mark.asyncio
    async def test_index_test_project(self, tmp_path):
        """Test indexing the test project."""
        available, msg = check_leansearch_available()
        if not available:
            pytest.skip(f"chromadb not installed: {msg}")

        # Use the existing test project
        test_project = Path(__file__).parent.parent / "test_project"
        if not test_project.exists():
            pytest.skip("test_project not found")

        mgr = LeanSearchManager(
            project_root=test_project,
            cache_dir=tmp_path / "cache",
        )

        # Index
        success = await mgr.ensure_indexed()
        assert success
        assert mgr.is_ready

        # Search for something that should exist
        results = mgr.search("hello", num_results=10)
        assert isinstance(results, list)

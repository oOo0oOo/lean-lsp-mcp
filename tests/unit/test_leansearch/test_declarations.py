"""Tests for leansearch.declarations module."""

import pytest
import tempfile
from pathlib import Path

pytest.importorskip("numpy", reason="leansearch tests require numpy")

from lean_lsp_mcp.leansearch.declarations import (
    compute_file_hash,
    infer_module_name,
    find_lean_files,
    extract_declarations_from_file,
)


class TestComputeFileHash:
    def test_hash_same_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write("theorem test : True := trivial")
            path1 = Path(f.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write("theorem test : True := trivial")
            path2 = Path(f.name)

        try:
            hash1 = compute_file_hash(path1)
            hash2 = compute_file_hash(path2)
            assert hash1 == hash2
        finally:
            path1.unlink()
            path2.unlink()

    def test_hash_different_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write("theorem test1 : True := trivial")
            path1 = Path(f.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write("theorem test2 : True := trivial")
            path2 = Path(f.name)

        try:
            hash1 = compute_file_hash(path1)
            hash2 = compute_file_hash(path2)
            assert hash1 != hash2
        finally:
            path1.unlink()
            path2.unlink()

    def test_nonexistent_file(self):
        result = compute_file_hash(Path("/nonexistent/file.lean"))
        assert result == ""


class TestInferModuleName:
    def test_mathlib_path(self):
        path = Path("/some/path/Mathlib/Algebra/Group/Basic.lean")
        result = infer_module_name(path)
        assert result == "Mathlib.Algebra.Group.Basic"

    def test_std_path(self):
        path = Path("/some/path/Std/Data/List/Basic.lean")
        result = infer_module_name(path)
        assert result == "Std.Data.List.Basic"

    def test_with_base_path(self):
        base = Path("/project")
        path = Path("/project/src/MyLib/Foo.lean")
        result = infer_module_name(path, base)
        # Should skip "src" prefix
        assert result == "MyLib.Foo"

    def test_simple_file(self):
        path = Path("/some/random/Basic.lean")
        result = infer_module_name(path)
        assert result == "Basic"


class TestFindLeanFiles:
    def test_find_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create some lean files
            (root / "test1.lean").write_text("-- test")
            (root / "subdir").mkdir()
            (root / "subdir" / "test2.lean").write_text("-- test")

            files = find_lean_files(root)
            assert len(files) == 2
            assert any("test1.lean" in str(f) for f in files)
            assert any("test2.lean" in str(f) for f in files)

    def test_exclude_build_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "test.lean").write_text("-- test")
            (root / ".lake" / "build").mkdir(parents=True)
            (root / ".lake" / "build" / "excluded.lean").write_text("-- excluded")

            files = find_lean_files(root, exclude_build=True)
            assert len(files) == 1
            assert "excluded.lean" not in str(files[0])

    def test_max_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for i in range(10):
                (root / f"test{i}.lean").write_text("-- test")

            files = find_lean_files(root, max_files=3)
            assert len(files) == 3


class TestExtractDeclarationsFromFile:
    def test_extract_theorem(self):
        content = """
theorem my_theorem : True := trivial
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            decls = extract_declarations_from_file(path)
            assert len(decls) == 1
            assert decls[0].kind == "theorem"
            assert "my_theorem" in decls[0].name
        finally:
            path.unlink()

    def test_extract_with_docstring(self):
        content = """
/-- This is a docstring -/
theorem documented_theorem : True := trivial
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            decls = extract_declarations_from_file(path)
            assert len(decls) == 1
            assert decls[0].docstring is not None
            assert "docstring" in decls[0].docstring
        finally:
            path.unlink()

    def test_extract_multiple_declarations(self):
        content = """
def my_def (x : Nat) : Nat := x + 1

lemma my_lemma : True := trivial

class MyClass where
  field : Nat

structure MyStruct where
  value : String
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            decls = extract_declarations_from_file(path)
            kinds = {d.kind for d in decls}
            assert "def" in kinds
            assert "lemma" in kinds
            assert "class" in kinds
            assert "structure" in kinds
        finally:
            path.unlink()

    def test_namespace_handling(self):
        content = """
namespace Foo

def bar : Nat := 42

namespace Inner

theorem baz : True := trivial

end Inner

end Foo
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            decls = extract_declarations_from_file(path)
            names = [d.name for d in decls]
            # Should have qualified names
            assert any("Foo" in n and "bar" in n for n in names)
            assert any("Inner" in n and "baz" in n for n in names)
        finally:
            path.unlink()

    def test_skip_private_names(self):
        content = """
def _private_def : Nat := 42
def public_def : Nat := 43
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            decls = extract_declarations_from_file(path)
            names = [d.name for d in decls]
            assert not any("_private_def" in n for n in names)
            assert any("public_def" in n for n in names)
        finally:
            path.unlink()

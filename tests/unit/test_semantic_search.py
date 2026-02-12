from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from lean_lsp_mcp import semantic_search


def test_extract_decls(tmp_path: Path) -> None:
    file_path = tmp_path / "Sample.lean"
    file_path.write_text(
        """
/-- doc -/\n
theorem foo : True := by\n  trivial\n
structure Bar where\n  x : Nat\n
def baz := 1\n""",
        encoding="utf-8",
    )

    items = list(semantic_search._extract_decls(file_path, tmp_path))
    names = [item.name for item in items]
    kinds = [item.kind for item in items]

    assert names == ["foo", "Bar", "baz"]
    assert kinds == ["theorem", "structure", "def"]


def test_index_prefix_stable(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    model = "sentence-transformers/all-MiniLM-L6-v2"
    prefix = semantic_search._index_prefix(root, model)
    assert prefix.count("-") == 1


def test_load_model_missing_dependency_shows_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise ModuleNotFoundError("No module named 'sentence_transformers'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    semantic_search._load_model.cache_clear()

    with pytest.raises(semantic_search.LeanToolError, match="uv add sentence-transformers numpy"):
        semantic_search._load_model("sentence-transformers/all-MiniLM-L6-v2")


def test_require_numpy_missing_dependency_shows_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            raise ModuleNotFoundError("No module named 'numpy'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(semantic_search.LeanToolError, match="uv add sentence-transformers numpy"):
        semantic_search._require_numpy()

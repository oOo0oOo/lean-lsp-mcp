from __future__ import annotations

from pathlib import Path

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

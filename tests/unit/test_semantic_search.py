from __future__ import annotations

import builtins
import os
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

    with pytest.raises(
        semantic_search.LeanToolError, match="uv add sentence-transformers numpy"
    ):
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

    with pytest.raises(
        semantic_search.LeanToolError, match="uv add sentence-transformers numpy"
    ):
        semantic_search._require_numpy()


def test_prepare_index_reuses_cached_embeddings_when_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    file_path = project / "Sample.lean"
    file_path.write_text("theorem foo : True := by\n  trivial\n", encoding="utf-8")

    stat = file_path.stat()
    rel = "Sample.lean"
    cached_item = semantic_search.SemanticSearchItem(
        name="foo",
        kind="theorem",
        file=rel,
        line=1,
        snippet="theorem foo : True := by",
    )
    cached_embeddings = object()
    cached_states = {
        rel: {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "content_hash": semantic_search._content_hash(
                file_path.read_text(encoding="utf-8")
            ),
        }
    }

    monkeypatch.setattr(
        semantic_search,
        "_load_index",
        lambda _index_dir, _prefix: ([cached_item], cached_embeddings, cached_states),
    )
    monkeypatch.setattr(
        semantic_search,
        "_load_model",
        lambda _model_name: (_ for _ in ()).throw(
            AssertionError("model should not be loaded when index is unchanged")
        ),
    )
    saved: list[tuple] = []
    monkeypatch.setattr(
        semantic_search,
        "_save_index",
        lambda *args, **kwargs: saved.append((args, kwargs)),
    )

    items, embeddings = semantic_search._prepare_index(
        project_root=project,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        rebuild=False,
    )

    assert items == [cached_item]
    assert embeddings is cached_embeddings
    assert saved == []


def test_prepare_index_uses_content_hash_for_touched_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    file_path = project / "Sample.lean"
    file_path.write_text("theorem foo : True := by\n  trivial\n", encoding="utf-8")

    stat_before = file_path.stat()
    rel = "Sample.lean"
    cached_item = semantic_search.SemanticSearchItem(
        name="foo",
        kind="theorem",
        file=rel,
        line=1,
        snippet="theorem foo : True := by",
    )
    cached_embeddings = object()
    content_hash = semantic_search._content_hash(file_path.read_text(encoding="utf-8"))
    cached_states = {
        rel: {
            "mtime_ns": stat_before.st_mtime_ns,
            "size": stat_before.st_size,
            "content_hash": content_hash,
        }
    }

    # Touch file without changing content.
    os.utime(
        file_path,
        ns=(stat_before.st_atime_ns + 1_000_000, stat_before.st_mtime_ns + 1_000_000),
    )

    monkeypatch.setattr(
        semantic_search,
        "_load_index",
        lambda _index_dir, _prefix: ([cached_item], cached_embeddings, cached_states),
    )
    monkeypatch.setattr(
        semantic_search,
        "_load_model",
        lambda _model_name: (_ for _ in ()).throw(
            AssertionError("model should not be loaded for touch-only updates")
        ),
    )
    saved: list[tuple] = []
    monkeypatch.setattr(
        semantic_search,
        "_save_index",
        lambda *args, **kwargs: saved.append((args, kwargs)),
    )

    items, embeddings = semantic_search._prepare_index(
        project_root=project,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        rebuild=False,
    )

    assert items == [cached_item]
    assert embeddings is cached_embeddings
    assert len(saved) == 1


class _FakeScores(list):
    def argsort(self):
        return sorted(range(len(self)), key=lambda idx: self[idx])


class _FakeEmbeddings:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def __matmul__(self, vector: list[float]) -> _FakeScores:
        return _FakeScores(
            [sum(left * right for left, right in zip(row, vector)) for row in self._rows]
        )


class _FakeNP:
    @staticmethod
    def array(value):
        return value

    @staticmethod
    def vstack(chunks):
        merged = []
        for chunk in chunks:
            merged.extend(chunk)
        return merged


def test_local_semantic_search_ranks_results(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [
        semantic_search.SemanticSearchItem(
            name="low", kind="def", file="A.lean", line=1, snippet="def low := 0"
        ),
        semantic_search.SemanticSearchItem(
            name="high", kind="def", file="B.lean", line=1, snippet="def high := 0"
        ),
        semantic_search.SemanticSearchItem(
            name="mid", kind="def", file="C.lean", line=1, snippet="def mid := 0"
        ),
    ]

    monkeypatch.setattr(
        semantic_search,
        "_prepare_index",
        lambda **_kwargs: (items, _FakeEmbeddings([[0.1], [0.9], [0.4]])),
    )
    monkeypatch.setattr(
        semantic_search,
        "_load_model",
        lambda _model_name: type(
            "_Model",
            (),
            {"encode": lambda self, _texts, normalize_embeddings=True: [[1.0]]},
        )(),
    )
    monkeypatch.setattr(semantic_search, "_require_numpy", lambda: _FakeNP)

    results = semantic_search.local_semantic_search(
        query="def",
        project_root=Path("/proj"),
        limit=2,
        model_name="fake-model",
        rebuild=False,
    )

    assert [item.name for item, _score in results] == ["high", "mid"]


def test_prepare_index_persists_and_reuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    file_path = project / "Sample.lean"
    file_path.write_text("def foo := 1\n", encoding="utf-8")
    stat = file_path.stat()

    monkeypatch.setattr(
        semantic_search,
        "_list_file_states",
        lambda _project_root: {"Sample.lean": (file_path, stat.st_mtime_ns, stat.st_size)},
    )

    store: dict[str, tuple[list[semantic_search.SemanticSearchItem], list[list[float]], dict]] = {}

    monkeypatch.setattr(
        semantic_search,
        "_load_index",
        lambda _index_dir, prefix: store.get(prefix),
    )

    monkeypatch.setattr(
        semantic_search,
        "_save_index",
        lambda _index_dir, prefix, items, embeddings, states: store.__setitem__(
            prefix, (items, embeddings, states)
        ),
    )

    class _Model:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, texts, normalize_embeddings=True):
            self.calls += 1
            return [[float(i + 1)] for i, _ in enumerate(texts)]

    model = _Model()
    monkeypatch.setattr(semantic_search, "_load_model", lambda _model_name: model)
    monkeypatch.setattr(semantic_search, "_require_numpy", lambda: _FakeNP)

    first_items, first_embeddings = semantic_search._prepare_index(
        project_root=project,
        model_name="fake-model",
        rebuild=False,
    )
    second_items, second_embeddings = semantic_search._prepare_index(
        project_root=project,
        model_name="fake-model",
        rebuild=False,
    )

    assert first_items
    assert model.calls == 1
    assert [item.name for item in second_items] == [item.name for item in first_items]
    assert second_embeddings == first_embeddings

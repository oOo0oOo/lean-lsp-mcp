"""Local semantic search using embeddings over Lean declarations."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from lean_lsp_mcp.search_utils import check_ripgrep_status
from lean_lsp_mcp.utils import LeanToolError

_DECL_PATTERN = re.compile(
    r"^\s*(theorem|lemma|def|axiom|class|instance|structure|inductive|abbrev|opaque)\s+([A-Za-z0-9_'.]+)"
)


@dataclass(frozen=True)
class SemanticSearchItem:
    name: str
    kind: str
    file: str
    line: int
    snippet: str


def get_cache_dir() -> Path:
    if d := os.environ.get("LEAN_SEMANTIC_SEARCH_CACHE_DIR"):
        return Path(d)
    xdg = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(xdg) / "lean-lsp-mcp" / "semantic-search"


def _hash_token(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:12]


def _content_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _index_prefix(project_root: Path, model_name: str) -> str:
    return f"{_hash_token(str(project_root))}-{_hash_token(model_name)}"


def _list_lean_files(project_root: Path) -> list[Path]:
    ok, msg = check_ripgrep_status()
    if not ok:
        raise LeanToolError(msg)

    command = [
        "rg",
        "--files",
        "--hidden",
        "-g",
        "*.lean",
        "-g",
        "!.git/**",
        "-g",
        "!.lake/build/**",
        str(project_root),
    ]

    result = subprocess.run(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise LeanToolError(result.stderr.strip() or "Failed to list Lean files")

    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _extract_decls_from_text(text: str, rel_file: str) -> list[SemanticSearchItem]:
    items: list[SemanticSearchItem] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        match = _DECL_PATTERN.match(line)
        if not match:
            continue
        kind, name = match.group(1), match.group(2)
        items.append(
            SemanticSearchItem(
                name=name,
                kind=kind,
                file=rel_file,
                line=idx,
                snippet=line.strip(),
            )
        )
    return items


def _extract_decls(file_path: Path, project_root: Path) -> Iterable[SemanticSearchItem]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return []

    try:
        rel = file_path.relative_to(project_root)
    except ValueError:
        rel = file_path
    return _extract_decls_from_text(text, str(rel))


def _list_file_states(project_root: Path) -> dict[str, tuple[Path, int, int]]:
    states: dict[str, tuple[Path, int, int]] = {}
    for file_path in _list_lean_files(project_root):
        try:
            stat = file_path.stat()
            rel = str(file_path.relative_to(project_root))
        except (FileNotFoundError, OSError, ValueError):
            continue
        states[rel] = (file_path, stat.st_mtime_ns, stat.st_size)
    return states


def _load_index(index_dir: Path, prefix: str):
    meta_path = index_dir / f"{prefix}.json"
    vec_path = index_dir / f"{prefix}.npy"
    if not meta_path.exists() or not vec_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    items = [SemanticSearchItem(**item) for item in meta.get("items", [])]

    file_states: dict[str, dict[str, int | str]] = {}
    raw_states = meta.get("file_states", {})
    if isinstance(raw_states, dict):
        for rel, value in raw_states.items():
            if not isinstance(rel, str) or not isinstance(value, dict):
                continue
            mtime_ns = int(value.get("mtime_ns", 0))
            size = int(value.get("size", 0))
            content_hash = str(value.get("content_hash", ""))
            file_states[rel] = {
                "mtime_ns": mtime_ns,
                "size": size,
                "content_hash": content_hash,
            }

    import numpy as np

    embeddings = np.load(vec_path)
    return items, embeddings, file_states


def _save_index(
    index_dir: Path,
    prefix: str,
    items,
    embeddings,
    file_states: dict[str, dict[str, int | str]],
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / f"{prefix}.json"
    vec_path = index_dir / f"{prefix}.npy"

    meta = {
        "items": [item.__dict__ for item in items],
        "file_states": file_states,
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    import numpy as np

    np.save(vec_path, embeddings)


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise LeanToolError(
            "Semantic search requires sentence-transformers. "
            "Install with `uv sync --extra semantic-search` or `uv add sentence-transformers numpy`."
        ) from exc

    return SentenceTransformer(model_name)


def _require_numpy():
    try:
        import numpy as np
    except Exception as exc:
        raise LeanToolError(
            "Semantic search requires numpy. "
            "Install with `uv sync --extra semantic-search` or `uv add sentence-transformers numpy`."
        ) from exc
    return np


def _prepare_index(
    *,
    project_root: Path,
    model_name: str,
    rebuild: bool,
):
    index_dir = get_cache_dir()
    prefix = _index_prefix(project_root, model_name)

    loaded = None if rebuild else _load_index(index_dir, prefix)
    if loaded is None:
        current_states = _list_file_states(project_root)
        changed_entries: list[tuple[str, str, int, int, str]] = []
        for rel, (file_path, mtime_ns, size) in current_states.items():
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            changed_entries.append((rel, text, mtime_ns, size, _content_hash(text)))

        if not changed_entries:
            return [], None

        model = _load_model(model_name)
        np = _require_numpy()

        new_items: list[SemanticSearchItem] = []
        for rel, text, *_ in changed_entries:
            new_items.extend(_extract_decls_from_text(text, rel))

        if not new_items:
            return [], None

        texts = [f"{item.name} {item.kind} {item.snippet}" for item in new_items]
        embeddings = np.array(model.encode(texts, normalize_embeddings=True))

        next_states = {
            rel: {"mtime_ns": mtime_ns, "size": size, "content_hash": content_hash}
            for rel, _text, mtime_ns, size, content_hash in changed_entries
        }
        _save_index(index_dir, prefix, new_items, embeddings, next_states)
        return new_items, embeddings

    items, embeddings, previous_states = loaded

    current_states = _list_file_states(project_root)
    removed_files = set(previous_states) - set(current_states)

    changed_entries: list[tuple[str, str, int, int, str]] = []
    next_states: dict[str, dict[str, int | str]] = {}
    touched_without_content_change = False

    for rel, (file_path, mtime_ns, size) in current_states.items():
        prev = previous_states.get(rel)
        if prev and prev.get("mtime_ns") == mtime_ns and prev.get("size") == size:
            next_states[rel] = prev
            continue

        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            removed_files.add(rel)
            continue

        content_hash = _content_hash(text)
        if prev and prev.get("content_hash") == content_hash:
            touched_without_content_change = True
            next_states[rel] = {
                "mtime_ns": mtime_ns,
                "size": size,
                "content_hash": content_hash,
            }
            continue

        changed_entries.append((rel, text, mtime_ns, size, content_hash))
        next_states[rel] = {
            "mtime_ns": mtime_ns,
            "size": size,
            "content_hash": content_hash,
        }

    if not removed_files and not changed_entries:
        if touched_without_content_change and items:
            _save_index(index_dir, prefix, items, embeddings, next_states)
        return items, embeddings

    drop_files = removed_files | {rel for rel, *_ in changed_entries}
    keep_indices = [idx for idx, item in enumerate(items) if item.file not in drop_files]
    kept_items = [items[idx] for idx in keep_indices]

    np = _require_numpy()
    if keep_indices:
        kept_embeddings = embeddings[keep_indices]
    else:
        kept_embeddings = None

    new_items: list[SemanticSearchItem] = []
    for rel, text, *_ in changed_entries:
        new_items.extend(_extract_decls_from_text(text, rel))

    if new_items:
        model = _load_model(model_name)
        new_texts = [f"{item.name} {item.kind} {item.snippet}" for item in new_items]
        new_embeddings = np.array(model.encode(new_texts, normalize_embeddings=True))

        if kept_embeddings is None:
            merged_embeddings = new_embeddings
        else:
            merged_embeddings = np.vstack([kept_embeddings, new_embeddings])
        merged_items = kept_items + new_items
    else:
        if kept_embeddings is None:
            return [], None
        merged_embeddings = kept_embeddings
        merged_items = kept_items

    _save_index(index_dir, prefix, merged_items, merged_embeddings, next_states)
    return merged_items, merged_embeddings


def local_semantic_search(
    *,
    query: str,
    project_root: Path,
    limit: int,
    model_name: str,
    rebuild: bool,
) -> list[tuple[SemanticSearchItem, float]]:
    items, embeddings = _prepare_index(
        project_root=project_root,
        model_name=model_name,
        rebuild=rebuild,
    )

    if not items or embeddings is None:
        return []

    model = _load_model(model_name)
    np = _require_numpy()
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec)[0]
    scores = embeddings @ query_vec
    top_idx = scores.argsort()[-limit:][::-1]

    results: list[tuple[SemanticSearchItem, float]] = []
    for idx in top_idx:
        results.append((items[int(idx)], float(scores[int(idx)])))

    return results

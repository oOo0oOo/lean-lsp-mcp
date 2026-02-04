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


def _extract_decls(file_path: Path, project_root: Path) -> Iterable[SemanticSearchItem]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return []

    items = []
    for idx, line in enumerate(text.splitlines(), start=1):
        match = _DECL_PATTERN.match(line)
        if not match:
            continue
        kind, name = match.group(1), match.group(2)
        try:
            rel = file_path.relative_to(project_root)
        except ValueError:
            rel = file_path
        items.append(
            SemanticSearchItem(
                name=name,
                kind=kind,
                file=str(rel),
                line=idx,
                snippet=line.strip(),
            )
        )
    return items


def _collect_items(project_root: Path) -> tuple[list[SemanticSearchItem], float]:
    files = _list_lean_files(project_root)
    items: list[SemanticSearchItem] = []
    latest_mtime = 0.0
    for file_path in files:
        try:
            stat = file_path.stat()
            latest_mtime = max(latest_mtime, stat.st_mtime)
        except FileNotFoundError:
            continue
        items.extend(_extract_decls(file_path, project_root))
    return items, latest_mtime


def _load_index(index_dir: Path, prefix: str):
    meta_path = index_dir / f"{prefix}.json"
    vec_path = index_dir / f"{prefix}.npy"
    if not meta_path.exists() or not vec_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    items = [SemanticSearchItem(**item) for item in meta["items"]]
    latest_mtime = meta["latest_mtime"]

    import numpy as np

    embeddings = np.load(vec_path)
    return items, embeddings, latest_mtime


def _save_index(index_dir: Path, prefix: str, items, embeddings, latest_mtime: float) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / f"{prefix}.json"
    vec_path = index_dir / f"{prefix}.npy"

    meta = {
        "latest_mtime": latest_mtime,
        "items": [item.__dict__ for item in items],
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
            "Install with `uv add --optional semantic-search sentence-transformers numpy`."
        ) from exc

    return SentenceTransformer(model_name)


def _require_numpy():
    try:
        import numpy as np
    except Exception as exc:
        raise LeanToolError(
            "Semantic search requires numpy. "
            "Install with `uv add --optional semantic-search sentence-transformers numpy`."
        ) from exc
    return np


def local_semantic_search(
    *,
    query: str,
    project_root: Path,
    limit: int,
    model_name: str,
    rebuild: bool,
) -> list[tuple[SemanticSearchItem, float]]:
    index_dir = get_cache_dir()
    prefix = _index_prefix(project_root, model_name)

    items, embeddings, latest_mtime = None, None, None

    if not rebuild:
        loaded = _load_index(index_dir, prefix)
        if loaded is not None:
            items, embeddings, latest_mtime = loaded

    current_items, current_mtime = _collect_items(project_root)
    if items is None or latest_mtime is None or current_mtime > latest_mtime:
        model = _load_model(model_name)
        np = _require_numpy()
        texts = [
            f"{item.name} {item.kind} {item.snippet}" for item in current_items
        ]
        if not texts:
            return []
        embeddings = model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings)
        _save_index(index_dir, prefix, current_items, embeddings, current_mtime)
        items = current_items

    if not items:
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

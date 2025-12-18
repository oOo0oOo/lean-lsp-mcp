"""Vector index and metadata storage for semantic search.

Uses usearch for fast HNSW vector search and SQLite for metadata storage.
This is lighter and faster than ChromaDB while maintaining filtering capability.

Architecture:
- usearch: Binary HNSW index for vector similarity search
- SQLite: Metadata storage with indexes for kind/module filtering
- Query flow: embed -> usearch.search() -> SQLite JOIN for metadata
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .models import LeanDeclaration

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Get the cache directory for local leansearch data."""
    if d := os.environ.get("LEAN_LEANSEARCH_CACHE_DIR"):
        return Path(d)
    xdg = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(xdg) / "lean-lsp-mcp" / "leansearch"


def compute_project_hash(project_root: Path) -> str:
    """Compute a hash of the project to detect changes requiring reindex."""
    hasher = hashlib.sha256()

    # Hash lake-manifest.json if it exists (tracks dependency versions)
    manifest = project_root / "lake-manifest.json"
    if manifest.exists():
        hasher.update(manifest.read_bytes())

    # Hash lakefile.lean or lakefile.toml
    for lakefile in ["lakefile.lean", "lakefile.toml"]:
        lf = project_root / lakefile
        if lf.exists():
            hasher.update(lf.read_bytes())
            break

    # Include project name for uniqueness
    hasher.update(project_root.name.encode())

    return hasher.hexdigest()[:16]


class LeanSearchIndex:
    """Vector index with SQLite metadata for semantic search.

    Combines usearch for fast vector search with SQLite for flexible
    metadata filtering. Supports incremental updates via file hashing.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        embedding_dim: int = 384,
        project_name: str = "default",
    ):
        self.cache_dir = cache_dir or get_cache_dir()
        self.embedding_dim = embedding_dim
        self.project_name = project_name

        # Paths
        project_dir = self.cache_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = project_dir / "vectors.usearch"
        self.db_path = project_dir / "metadata.db"
        self.hashes_path = project_dir / "file_hashes.json"

        # State
        self._index = None
        self._conn: sqlite3.Connection | None = None
        self._file_hashes: dict[str, str] = {}

    def _get_index(self):
        """Get or create the usearch index."""
        if self._index is None:
            try:
                from usearch.index import Index
            except ImportError:
                raise RuntimeError(
                    "usearch is required for local leansearch. "
                    "Install with: pip install usearch"
                )

            self._index = Index(ndim=self.embedding_dim, metric="cos")

            # Load existing index if available
            if self.index_path.exists():
                try:
                    self._index.load(str(self.index_path))
                    logger.info(f"Loaded index with {len(self._index)} vectors")
                except Exception as e:
                    logger.warning(f"Could not load index: {e}")

        return self._index

    def _get_db(self) -> sqlite3.Connection:
        """Get or create the SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._init_db()
        return self._conn

    def _init_db(self) -> None:
        """Create SQLite schema for declaration metadata."""
        conn = self._conn
        if conn is None:
            return

        conn.execute("""
            CREATE TABLE IF NOT EXISTS declarations (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                kind TEXT,
                module TEXT,
                signature TEXT,
                docstring TEXT,
                file_path TEXT,
                line INTEGER,
                search_text TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kind ON declarations(kind)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_module ON declarations(module)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_path ON declarations(file_path)"
        )
        conn.commit()

    def load_file_hashes(self) -> dict[str, str]:
        """Load cached file hashes from disk."""
        if self.hashes_path.exists():
            try:
                self._file_hashes = json.loads(self.hashes_path.read_text())
                return self._file_hashes
            except Exception:
                pass
        self._file_hashes = {}
        return self._file_hashes

    def save_file_hashes(self) -> None:
        """Save file hashes to disk."""
        self.hashes_path.write_text(json.dumps(self._file_hashes))

    def count(self) -> int:
        """Get number of declarations in the index."""
        conn = self._get_db()
        result = conn.execute("SELECT COUNT(*) FROM declarations").fetchone()
        return result[0] if result else 0

    def add(
        self,
        declarations: list[LeanDeclaration],
        embed_fn: Callable[[list[str]], np.ndarray],
        batch_size: int = 500,
    ) -> None:
        """Add declarations with their embeddings.

        Args:
            declarations: List of declarations to add
            embed_fn: Function to compute embeddings for a batch of texts
            batch_size: Number of declarations to process at once
        """
        if not declarations:
            return

        index = self._get_index()
        conn = self._get_db()

        # Process in batches
        for i in range(0, len(declarations), batch_size):
            batch = declarations[i : i + batch_size]

            # Build search texts with docstring weighting
            search_texts = []
            for decl in batch:
                parts = [decl.name.replace(".", " "), decl.kind]
                if decl.docstring:
                    # Weight docstrings heavily - they contain natural language
                    parts.extend([decl.docstring] * 3)
                if decl.signature:
                    parts.append(decl.signature)
                search_texts.append(" ".join(parts))

            # Compute embeddings
            embeddings = embed_fn(search_texts)

            # Insert into SQLite and usearch
            new_ids = []
            new_embeddings = []

            for j, decl in enumerate(batch):
                # Check if declaration already exists
                existing = conn.execute(
                    "SELECT id FROM declarations WHERE name = ?", (decl.name,)
                ).fetchone()

                if existing:
                    # Update existing record, don't re-add to vector index
                    conn.execute(
                        """
                        UPDATE declarations SET
                            kind=?, module=?, signature=?, docstring=?,
                            file_path=?, line=?, search_text=?
                        WHERE name=?
                        """,
                        (
                            decl.kind,
                            decl.module,
                            decl.signature[:500] if decl.signature else "",
                            decl.docstring[:1000] if decl.docstring else None,
                            decl.file_path,
                            decl.line,
                            search_texts[j],
                            decl.name,
                        ),
                    )
                else:
                    # Insert new record
                    cursor = conn.execute(
                        """
                        INSERT INTO declarations (name, kind, module, signature, docstring, file_path, line, search_text)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        RETURNING id
                        """,
                        (
                            decl.name,
                            decl.kind,
                            decl.module,
                            decl.signature[:500] if decl.signature else "",
                            decl.docstring[:1000] if decl.docstring else None,
                            decl.file_path,
                            decl.line,
                            search_texts[j],
                        ),
                    )
                    row = cursor.fetchone()
                    if row:
                        new_ids.append(row[0])
                        new_embeddings.append(embeddings[j])

            # Batch add new vectors to index
            if new_ids:
                for decl_id, emb in zip(new_ids, new_embeddings):
                    index.add(decl_id, emb)

            conn.commit()

            logger.debug(
                f"Indexed batch {i // batch_size + 1}: {len(batch)} declarations"
            )

        # Save index to disk
        index.save(str(self.index_path))

    def remove_by_file(self, file_path: str) -> int:
        """Remove all declarations from a file.

        Returns number of declarations removed.
        """
        conn = self._get_db()

        # Get IDs to remove from vector index
        rows = conn.execute(
            "SELECT id FROM declarations WHERE file_path = ?", (file_path,)
        ).fetchall()

        if not rows:
            return 0

        ids = [row[0] for row in rows]

        # Note: usearch doesn't support deletion, so we'll need to rebuild
        # For now, just remove from SQLite - rebuild handles vector index
        conn.execute("DELETE FROM declarations WHERE file_path = ?", (file_path,))
        conn.commit()

        return len(ids)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        kind: str | None = None,
        module: str | None = None,
        overfetch_factor: int = 3,
    ) -> list[dict[str, Any]]:
        """Search with optional metadata filtering.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            kind: Filter by declaration kind (theorem, lemma, def, etc.)
            module: Filter by module prefix
            overfetch_factor: How many extra results to fetch for filtering

        Returns:
            List of matching declarations with metadata and distances
        """
        index = self._get_index()
        conn = self._get_db()

        if len(index) == 0:
            return []

        # Overfetch for post-filtering
        fetch_k = min(k * overfetch_factor, len(index))
        matches = index.search(query_embedding, fetch_k)

        if len(matches) == 0:
            return []

        # Get IDs and distances
        ids = [int(m.key) for m in matches]
        distances = [float(m.distance) for m in matches]

        # Build SQL query with optional filters
        placeholders = ",".join("?" * len(ids))
        sql = f"SELECT * FROM declarations WHERE id IN ({placeholders})"
        params: list[Any] = list(ids)

        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if module:
            sql += " AND module LIKE ?"
            params.append(f"{module}%")

        rows = conn.execute(sql, params).fetchall()

        # Build results with distances
        id_to_distance = dict(zip(ids, distances))
        results = []
        for row in rows:
            result = dict(row)
            result["distance"] = id_to_distance.get(row["id"], 1.0)
            results.append(result)

        # Sort by distance and limit
        results.sort(key=lambda x: x["distance"])
        return results[:k]

    def search_by_text(
        self,
        query: str,
        embed_fn: Callable[[list[str]], np.ndarray],
        k: int = 10,
        kind: str | None = None,
        module: str | None = None,
        hybrid_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search by text query with hybrid semantic + keyword matching.

        Args:
            query: Natural language search query
            embed_fn: Embedding function
            k: Number of results
            kind: Filter by kind
            module: Filter by module prefix
            hybrid_weight: Weight for keyword score (0-1)

        Returns:
            List of matching declarations
        """
        import re

        query_embedding = embed_fn([query])[0]
        results = self.search(query_embedding, k * 3, kind, module)

        # Detect if query looks like an identifier (single word, camelCase, or dotted)
        query_stripped = query.strip()
        is_identifier_query = (
            len(query_stripped.split()) == 1
            and not query_stripped.startswith('"')
            and len(query_stripped) >= 3
        )

        # For identifier queries, also search by name in SQL to catch exact matches
        # that might not have high semantic similarity
        if is_identifier_query:
            conn = self._get_db()
            sql = """
                SELECT * FROM declarations
                WHERE LOWER(name) LIKE ?
            """
            params = [f"%{query_stripped.lower()}%"]

            if kind:
                sql += " AND kind = ?"
                params.append(kind)
            if module:
                sql += " AND module LIKE ?"
                params.append(f"{module}%")

            sql += " LIMIT ?"
            params.append(k * 3)

            sql_results = conn.execute(sql, params).fetchall()

            # Add SQL matches not already in results, with distance based on match quality
            seen_ids = {r.get("id") for r in results}
            query_lower = query_stripped.lower()
            for row in sql_results:
                row_dict = dict(row)
                if row_dict["id"] not in seen_ids:
                    # Compute match-quality distance for SQL results
                    name = row_dict.get("name", "")
                    name_last = name.split(".")[-1].lower() if name else ""

                    if name_last == query_lower:
                        # Exact match on final component - very strong
                        row_dict["distance"] = 0.05
                    elif name_last.endswith(query_lower):
                        # Suffix match (e.g., "instToJson" for "toJson") - strong
                        row_dict["distance"] = 0.15
                    elif name_last.startswith(query_lower):
                        # Prefix match - moderately strong
                        row_dict["distance"] = 0.25
                    elif query_lower in name_last:
                        # Substring match - decent
                        row_dict["distance"] = 0.35
                    else:
                        # Match elsewhere in qualified name
                        row_dict["distance"] = 0.45

                    results.append(row_dict)
                    seen_ids.add(row_dict["id"])

        if not results:
            return []

        # For identifier queries, boost keyword weight significantly
        effective_weight = 0.7 if is_identifier_query else hybrid_weight

        # Normalize query for matching
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # Split camelCase for additional matching
        camel_parts = set()
        for term in query_terms:
            # Split on camelCase boundaries
            parts = re.findall(r"[a-z]+|[A-Z][a-z]*", term)
            camel_parts.update(p.lower() for p in parts if len(p) > 1)
        query_terms.update(camel_parts)

        scored = []
        for r in results:
            name = r.get("name", "")
            text = (r.get("search_text") or name).lower()

            # Base keyword score
            kw_matches = sum(1 for t in query_terms if t in text)
            kw_score = kw_matches / len(query_terms) if query_terms else 0

            # Bonus for exact name component match (e.g., "toJson" matches "instToJson")
            name_parts = name.split(".")
            exact_bonus = 0.0
            for part in name_parts:
                part_lower = part.lower()
                if query_lower in part_lower:
                    # Substring match in name component
                    exact_bonus = max(exact_bonus, 0.3)
                if part_lower == query_lower:
                    # Exact match on a name component - strongest
                    exact_bonus = max(exact_bonus, 0.7)
                elif part_lower.endswith(query_lower):
                    # Suffix match (e.g., "instToJson" for "toJson") - very strong
                    # This catches Lean's `inst*` naming convention
                    exact_bonus = max(exact_bonus, 0.6)
                elif part_lower.startswith(query_lower):
                    # Prefix match - strong
                    exact_bonus = max(exact_bonus, 0.5)

            # Semantic score (lower distance = better)
            semantic_score = 1.0 - min(r["distance"], 1.0)

            # Combined score
            keyword_total = min(1.0, kw_score + exact_bonus)
            hybrid = (
                1 - effective_weight
            ) * semantic_score + effective_weight * keyword_total
            r["hybrid_score"] = hybrid
            scored.append(r)

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored[:k]

    def clear(self) -> None:
        """Clear all data from the index."""
        # Ensure we have a connection to clear the database
        conn = self._get_db()
        conn.execute("DELETE FROM declarations")
        conn.commit()

        # Remove usearch index file
        if self.index_path.exists():
            self.index_path.unlink()
        self._index = None

        self._file_hashes = {}
        if self.hashes_path.exists():
            self.hashes_path.unlink()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

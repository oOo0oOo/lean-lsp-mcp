"""Local semantic search for Lean 4 projects using embeddings.

This package provides local semantic search functionality similar to leansearch.net
but runs entirely locally with no rate limits. It indexes Lean declarations from
your project and dependencies, then uses vector embeddings for semantic search.

Features:
- Hybrid search: semantic + keyword matching for better results
- Incremental indexing: only reindex changed files
- Docstring weighting: prioritizes natural language descriptions
- Parallel extraction: uses all CPU cores for fast indexing
- Premise-based search: uses dependency graph for goal-aware results

Architecture:
- usearch: Fast HNSW vector index (replaces ChromaDB)
- SQLite: Metadata storage with filtering support
- sentence-transformers: Local embeddings (no API key needed)
- lean-training-data: Optional accurate extraction for mathlib

Example:
    from lean_lsp_mcp.leansearch import LeanSearchManager

    manager = LeanSearchManager(project_root=Path("/path/to/lean/project"))
    await manager.ensure_indexed()
    results = manager.search("sum of two even numbers is even", num_results=5)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .models import LeanDeclaration, IndexStats, PremiseGraph
from .embeddings import get_embedding_function, check_embeddings_available
from .declarations import (
    find_lean_files,
    extract_declarations_from_file,
    compute_file_hash,
)
from .indexer import LeanSearchIndex, get_cache_dir, compute_project_hash
from .premises import LocalPremiseSearch
from .training_data import TrainingDataExtractor, find_training_data_repo

__all__ = [
    "LeanSearchManager",
    "LocalPremiseSearch",
    "LeanDeclaration",
    "IndexStats",
    "PremiseGraph",
    "check_leansearch_available",
]

logger = logging.getLogger(__name__)


def check_leansearch_available() -> tuple[bool, str]:
    """Check if local leansearch dependencies are available."""
    # Check usearch
    try:
        import usearch  # noqa: F401
    except ImportError:
        return False, (
            "usearch is required for local semantic search. "
            "Install with: pip install usearch"
        )

    # Check embeddings
    return check_embeddings_available()


class LeanSearchManager:
    """Manages local semantic search for Lean 4 projects.

    Features:
    - Hybrid search: semantic + keyword matching
    - Incremental indexing: only reindex changed files
    - Parallel extraction: uses all CPU cores
    - Multiple embedding backends (local or API-based)
    - Caches indices per project version
    - No rate limits!
    """

    def __init__(
        self,
        project_root: Path | None = None,
        cache_dir: Path | None = None,
        embedding_provider: str = "default",
        embedding_model: str | None = None,
        training_data_path: Path | None = None,
    ):
        """Initialize the search manager.

        Args:
            project_root: Root of Lean project to index
            cache_dir: Directory for caching index data
            embedding_provider: One of 'default', 'openai', 'gemini', 'voyage'
            embedding_model: Override default model for provider
            training_data_path: Path to lean-training-data repo (optional)
        """
        self.project_root = project_root
        self.cache_dir = cache_dir or get_cache_dir()
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.training_data_path = training_data_path or find_training_data_repo()

        self._index: LeanSearchIndex | None = None
        self._embed_fn: Callable[[list[str]], Any] | None = None
        self._embed_dim: int = 384
        self._ready = False
        self._lock = asyncio.Lock()
        self._stats: IndexStats | None = None
        self._premise_graph: PremiseGraph | None = None
        self._premise_search: LocalPremiseSearch | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def stats(self) -> IndexStats | None:
        return self._stats

    def _get_project_name(self) -> str:
        """Get a unique name for this project's index."""
        if self.project_root:
            project_hash = compute_project_hash(self.project_root)
            return f"{self.project_root.name}_{project_hash}"
        return "global"

    def _get_embed_fn(self) -> tuple[Callable[[list[str]], Any], int]:
        """Get or create embedding function."""
        if self._embed_fn is None:
            self._embed_fn, self._embed_dim = get_embedding_function(
                self.embedding_provider, self.embedding_model
            )
            # Warm up the model with a dummy query to avoid first-query latency
            try:
                self._embed_fn(["warmup query"])
            except Exception:
                pass
        return self._embed_fn, self._embed_dim

    def _get_index(self) -> LeanSearchIndex:
        """Get or create the search index."""
        if self._index is None:
            _, dim = self._get_embed_fn()
            self._index = LeanSearchIndex(
                cache_dir=self.cache_dir,
                embedding_dim=dim,
                project_name=self._get_project_name(),
            )
        return self._index

    def _collect_lean_files(
        self, progress_callback: Callable[[str, int, int], None] | None = None
    ) -> list[tuple[Path, str, Path]]:
        """Collect all Lean files to index with their module prefixes and base paths.

        Returns list of (file_path, module_prefix, base_path) tuples.
        Prefers main project files over .lake/packages duplicates.
        """
        files: list[tuple[Path, str, Path]] = []
        seen_relative_paths: set[str] = set()

        if not self.project_root:
            return files

        if progress_callback:
            progress_callback("Scanning project files...", 0, 1)

        # Project source files (highest priority)
        for src_dir in [".", "src", self.project_root.name]:
            src_path = self.project_root / src_dir
            if src_path.exists() and src_path.is_dir():
                lean_files = find_lean_files(src_path)
                for f in lean_files:
                    # Track relative path to avoid duplicates
                    try:
                        rel = f.relative_to(src_path)
                        rel_str = str(rel)
                    except ValueError:
                        rel_str = str(f)

                    if rel_str not in seen_relative_paths:
                        files.append((f, "", src_path))
                        seen_relative_paths.add(rel_str)

                if lean_files:
                    logger.info(f"Found {len(lean_files)} files in {src_path}")

        # .lake/packages dependencies (lower priority, skip duplicates)
        lake_packages = self.project_root / ".lake" / "packages"
        if lake_packages.exists():
            pkg_dirs = [d for d in lake_packages.iterdir() if d.is_dir()]
            for idx, pkg_dir in enumerate(pkg_dirs):
                if progress_callback:
                    progress_callback(f"Scanning {pkg_dir.name}...", idx, len(pkg_dirs))

                # Look for source files in common locations
                for subdir in [".", "src", pkg_dir.name]:
                    sub_path = pkg_dir / subdir
                    if sub_path.exists() and sub_path.is_dir():
                        pkg_files = find_lean_files(sub_path)
                        for f in pkg_files:
                            # Check if this file's relative path was already seen
                            try:
                                rel = f.relative_to(sub_path)
                                rel_str = str(rel)
                            except ValueError:
                                rel_str = str(f)

                            if rel_str not in seen_relative_paths:
                                files.append((f, "", sub_path))
                                seen_relative_paths.add(rel_str)

        return files

    def index_project(
        self,
        force: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> int:
        """Index the project and dependencies with incremental updates.

        Uses parallel extraction and only reindexes changed files.

        Args:
            force: If True, reindex all files even if unchanged
            progress_callback: Optional callback(message, current, total) for progress

        Returns:
            Number of declarations indexed
        """
        start_time = time.monotonic()
        index = self._get_index()
        embed_fn, _ = self._get_embed_fn()

        # Load cached file hashes for incremental indexing
        cached_hashes = {} if force else index.load_file_hashes()

        # Check if already indexed (and not forcing)
        if not force and index.count() > 0 and cached_hashes:
            logger.info(f"Index already has {index.count()} items, skipping")
            self._ready = True
            return index.count()

        # Clear if forcing
        if force:
            logger.info("Forcing reindex, clearing existing data...")
            index.clear()
            cached_hashes = {}

        if progress_callback:
            progress_callback("Starting indexing...", 0, 100)

        logger.info("Starting project indexing...")
        files = self._collect_lean_files(progress_callback)
        total_files = len(files)
        logger.info(f"Found {total_files} Lean files to index")

        # Determine which files need processing (incremental)
        files_to_process: list[tuple[Path, str, Path]] = []
        new_hashes: dict[str, str] = {}
        files_added = 0
        files_updated = 0
        files_unchanged = 0

        for file_path, module, base_path in files:
            file_key = str(file_path)
            current_hash = compute_file_hash(file_path)
            new_hashes[file_key] = current_hash

            if file_key not in cached_hashes:
                files_to_process.append((file_path, module, base_path))
                files_added += 1
            elif cached_hashes[file_key] != current_hash:
                files_to_process.append((file_path, module, base_path))
                files_updated += 1
            else:
                files_unchanged += 1

        logger.info(
            f"Incremental index: {files_added} new, {files_updated} changed, "
            f"{files_unchanged} unchanged"
        )

        if not files_to_process and index.count() > 0:
            logger.info("No files changed, index is up to date")
            self._ready = True
            index._file_hashes = new_hashes
            index.save_file_hashes()
            return index.count()

        if progress_callback:
            progress_callback(
                f"Extracting from {len(files_to_process)} files...",
                0,
                len(files_to_process),
            )

        # Parallel extraction using ThreadPoolExecutor
        all_declarations: list[LeanDeclaration] = []
        kind_counts: dict[str, int] = {}
        max_workers = min(os.cpu_count() or 4, 8)

        def extract_file(args: tuple[Path, str, Path]) -> list[LeanDeclaration]:
            file_path, module, base_path = args
            return extract_declarations_from_file(file_path, module, base_path)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(extract_file, f): i
                for i, f in enumerate(files_to_process)
            }

            completed = 0
            for future in as_completed(futures):
                try:
                    decls = future.result()
                    for d in decls:
                        kind_counts[d.kind] = kind_counts.get(d.kind, 0) + 1
                    all_declarations.extend(decls)
                except Exception as e:
                    logger.warning(f"Failed to extract file: {e}")

                completed += 1
                if progress_callback and completed % 100 == 0:
                    progress_callback(
                        f"Extracted {len(all_declarations)} declarations...",
                        completed,
                        len(files_to_process),
                    )

        if not all_declarations:
            logger.warning("No declarations found to index")
            self._ready = True
            index._file_hashes = new_hashes
            index.save_file_hashes()
            self._stats = IndexStats(
                total_declarations=0,
                total_files=total_files,
                project_name=self.project_root.name if self.project_root else "",
                embedding_provider=self.embedding_provider,
                files_added=files_added,
                files_updated=files_updated,
                files_unchanged=files_unchanged,
            )
            return 0

        logger.info(f"Extracted {len(all_declarations)} declarations")

        if progress_callback:
            progress_callback(
                f"Building embeddings for {len(all_declarations)} declarations...",
                0,
                len(all_declarations),
            )

        # Add to index
        index.add(all_declarations, embed_fn)

        elapsed = time.monotonic() - start_time
        self._ready = True

        # Save file hashes
        index._file_hashes = new_hashes
        index.save_file_hashes()

        self._stats = IndexStats(
            total_declarations=index.count(),
            total_files=total_files,
            declarations_by_kind=kind_counts,
            index_time_seconds=elapsed,
            project_name=self.project_root.name if self.project_root else "",
            embedding_provider=self.embedding_provider,
            files_added=files_added,
            files_updated=files_updated,
            files_unchanged=files_unchanged,
        )

        logger.info(
            f"Indexing complete: {index.count()} declarations in {elapsed:.1f}s "
            f"({files_added} new, {files_updated} updated, {files_unchanged} unchanged)"
        )
        return index.count()

    def search(
        self, query: str, num_results: int = 5, hybrid_weight: float = 0.3
    ) -> list[dict[str, Any]]:
        """Search for declarations using hybrid semantic + keyword matching.

        Args:
            query: Natural language search query
            num_results: Maximum number of results
            hybrid_weight: Weight for keyword score (0-1). Higher = more keyword influence.

        Returns:
            List of matching declarations with metadata and similarity scores
        """
        if not self._ready:
            raise RuntimeError("Index not ready. Call index_project() first.")

        index = self._get_index()
        embed_fn, _ = self._get_embed_fn()

        return index.search_by_text(
            query, embed_fn, k=num_results, hybrid_weight=hybrid_weight
        )

    def _goal_to_loogle_query(self, goal_state: str) -> str | None:
        """Convert a goal state to a loogle query pattern.

        Examples:
            "⊢ List.length (List.map f xs) = List.length xs"
            -> "List.length (List.map _ _) = List.length _"

            "h : x ∈ xs ⊢ List.find? (· == x) xs = some x"
            -> "List.find? _ _ = some _"
        """
        import re

        # Extract the target (after ⊢)
        if "⊢" in goal_state:
            target = goal_state.split("⊢")[-1].strip()
        else:
            target = goal_state.strip()

        if not target:
            return None

        # Strategy: Keep qualified names (Foo.bar), replace standalone variables
        pattern = target

        # First, protect qualified names by marking them
        # Match patterns like List.length, Option.some, etc.
        qualified_names: list[str] = []

        def protect_qualified(m: re.Match) -> str:
            qualified_names.append(m.group(0))
            return f"__QUAL{len(qualified_names) - 1}__"

        # Protect qualified names (capitalized.anything or anything.anything.etc)
        pattern = re.sub(
            r"\b([A-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_?']+)+)\b",
            protect_qualified,
            pattern,
        )

        # Replace standalone lowercase variables (single word, not part of qualified)
        # But keep common Lean operators/keywords
        keep_words = {
            "true",
            "false",
            "some",
            "none",
            "fun",
            "let",
            "if",
            "then",
            "else",
        }

        def replace_var(m: re.Match) -> str:
            word = m.group(1)
            if word in keep_words:
                return word
            return "_"

        pattern = re.sub(r"\b([a-z][a-z0-9_']*)\b", replace_var, pattern)

        # Restore qualified names
        for i, name in enumerate(qualified_names):
            pattern = pattern.replace(f"__QUAL{i}__", name)

        # Clean up multiple underscores and spaces
        pattern = re.sub(r"_\s+_", "_ _", pattern)
        pattern = re.sub(r"\s+", " ", pattern)
        pattern = pattern.strip()

        # If the pattern has only underscores (no structure), return None
        # But keep patterns with operators - they can still be useful for loogle
        has_structure = bool(
            re.search(r"[A-Z][a-zA-Z0-9_]*", pattern)  # Has a capitalized name
            or re.search(r"\d+", pattern)  # Has a number
            or ("=" in pattern and pattern.count("_") >= 2)  # Equation with wildcards
        )

        if not has_structure and re.fullmatch(r"[_\s]+", pattern):
            return None

        return pattern

    def search_by_goal(
        self, goal_state: str, num_results: int = 10, use_loogle: bool = True
    ) -> list[dict[str, Any]]:
        """Find lemmas relevant to a goal state.

        Uses a hybrid approach:
        1. Loogle (remote API) for mathlib pattern matching
        2. Local semantic search for project-specific lemmas
        3. Premise graph if available

        Args:
            goal_state: The goal state text
            num_results: Number of results
            use_loogle: Whether to query loogle for mathlib results

        Returns:
            List of relevant declarations
        """
        results: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        # 1. Try loogle for mathlib lemmas (type pattern matching)
        if use_loogle:
            loogle_query = self._goal_to_loogle_query(goal_state)
            if loogle_query:
                try:
                    from ..loogle import loogle_remote

                    loogle_results = loogle_remote(loogle_query, num_results)
                    if isinstance(loogle_results, list):
                        for i, r in enumerate(loogle_results):
                            name = r.get("name", "")
                            if name and name not in seen_names:
                                results.append(
                                    {
                                        "name": name,
                                        "kind": r.get("kind", "theorem"),
                                        "signature": r.get("type", ""),
                                        "module": r.get("module", ""),
                                        "score": 1.0
                                        - (i * 0.05),  # Rank by loogle order
                                        "source": "loogle",
                                    }
                                )
                                seen_names.add(name)
                except Exception as e:
                    logger.debug(f"Loogle query failed: {e}")

        # 2. Use premise search if available
        if self._ready and self._premise_search is not None:
            embed_fn, _ = self._get_embed_fn()
            try:
                premise_results = self._premise_search.search_by_goal(
                    goal_state, embed_fn, num_results
                )
                for r in premise_results:
                    name = r.get("name", "")
                    if name and name not in seen_names:
                        r["source"] = "premise"
                        results.append(r)
                        seen_names.add(name)
            except Exception as e:
                logger.debug(f"Premise search failed: {e}")

        # 3. Fall back to semantic search on local index
        if self._ready:
            index = self._get_index()
            embed_fn, _ = self._get_embed_fn()

            # Extract target for semantic search
            if "⊢" in goal_state:
                target = goal_state.split("⊢")[-1].strip()
            else:
                target = goal_state

            try:
                semantic_results = index.search_by_text(target, embed_fn, k=num_results)
                for r in semantic_results:
                    name = r.get("name", "")
                    if name and name not in seen_names:
                        r["source"] = "semantic"
                        # Normalize score key
                        if "hybrid_score" in r:
                            r["score"] = r["hybrid_score"]
                        elif "distance" in r:
                            r["score"] = 1.0 - min(r["distance"], 1.0)
                        results.append(r)
                        seen_names.add(name)
            except Exception as e:
                logger.debug(f"Semantic search failed: {e}")

        # Sort by score and return top results
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:num_results]

    async def ensure_indexed(self, project_root: Path | None = None) -> bool:
        """Ensure the project is indexed, indexing if necessary.

        This is the main entry point for lazy initialization.
        Thread-safe via asyncio lock.
        """
        async with self._lock:
            if project_root and project_root != self.project_root:
                # Project changed, need to reindex
                self.project_root = project_root
                self._index = None
                self._ready = False

            if self._ready:
                return True

            try:
                count = self.index_project()
                return count >= 0
            except Exception as e:
                logger.error(f"Indexing failed: {e}")
                return False

    def load_premise_graph(self) -> bool:
        """Load premise graph from lean-training-data if available.

        Returns True if successful.
        """
        if self.training_data_path is None:
            return False

        extractor = TrainingDataExtractor(self.training_data_path)
        if not extractor.is_available():
            return False

        try:
            self._premise_graph = extractor.extract_premises()
            if self._premise_graph and self._premise_graph.adjacency:
                index = self._get_index()
                self._premise_search = LocalPremiseSearch(self._premise_graph, index)
                logger.info(
                    f"Loaded premise graph with {len(self._premise_graph.adjacency)} nodes"
                )
                return True
        except Exception as e:
            logger.warning(f"Failed to load premise graph: {e}")

        return False

    def reindex(
        self, progress_callback: Callable[[str, int, int], None] | None = None
    ) -> int:
        """Force reindex the project."""
        return self.index_project(force=True, progress_callback=progress_callback)

    def clear_cache(self) -> None:
        """Clear the cached index."""
        if self._index is not None:
            self._index.clear()
            self._index = None
        self._ready = False
        self._stats = None

    def close(self) -> None:
        """Close resources."""
        if self._index is not None:
            self._index.close()

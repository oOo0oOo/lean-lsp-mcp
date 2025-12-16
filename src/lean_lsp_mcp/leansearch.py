"""Local semantic search for Lean 4 projects using embeddings.

This module provides local semantic search functionality similar to leansearch.net
but runs entirely locally with no rate limits. It indexes Lean declarations from
your project and dependencies, then uses vector embeddings for semantic search.

Features:
- Hybrid search: semantic + keyword matching for better results
- Incremental indexing: only reindex changed files
- Docstring weighting: prioritizes natural language descriptions
- Parallel extraction: uses all CPU cores for fast indexing

Supports multiple embedding backends:
- sentence-transformers (default, no API key required)
- OpenAI text-embedding-3-large (best quality)
- Google text-embedding-004 (good quality, free tier)
- Voyage AI voyage-code-2 (excellent for code)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class LeanDeclaration:
    """A Lean 4 declaration extracted from source."""

    name: str
    kind: str  # theorem, lemma, def, class, structure, etc.
    module: str  # Module path e.g. Mathlib.Algebra.Group
    signature: str  # Type signature
    docstring: str | None
    file_path: str
    line: int


@dataclass
class IndexStats:
    """Statistics about the indexed project."""

    total_declarations: int = 0
    total_files: int = 0
    declarations_by_kind: dict[str, int] = field(default_factory=dict)
    index_time_seconds: float = 0.0
    project_name: str = ""
    embedding_provider: str = ""
    # Incremental indexing stats
    files_added: int = 0
    files_updated: int = 0
    files_unchanged: int = 0


def _compute_file_hash(file_path: Path) -> str:
    """Compute hash of file contents for change detection."""
    try:
        return hashlib.md5(file_path.read_bytes()).hexdigest()[:12]
    except Exception:
        return ""


def _keyword_score(query: str, text: str) -> float:
    """Compute keyword overlap score for hybrid search.

    Returns a score between 0 and 1 based on query term matches.
    """
    query_terms = set(query.lower().split())
    text_lower = text.lower()

    if not query_terms:
        return 0.0

    matches = sum(1 for term in query_terms if term in text_lower)
    return matches / len(query_terms)


def get_cache_dir() -> Path:
    """Get the cache directory for local leansearch data."""
    if d := os.environ.get("LEAN_LEANSEARCH_CACHE_DIR"):
        return Path(d)
    xdg = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(xdg) / "lean-lsp-mcp" / "leansearch"


def _compute_project_hash(project_root: Path) -> str:
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


def _infer_module_name(file_path: Path, base_path: Path | None = None) -> str:
    """Infer module name from file path.

    Handles common Lean project structures:
    - Mathlib/Algebra/Group/Basic.lean -> Mathlib.Algebra.Group.Basic
    - src/MyLib/Foo.lean -> MyLib.Foo
    - Basic.lean -> Basic
    """
    parts = file_path.parts

    # Try to find a recognizable root
    roots = {"Mathlib", "Std", "Init", "Lean", "Batteries", "Aesop", "ProofWidgets"}

    for i, part in enumerate(parts):
        if part in roots:
            # Found a known library root
            rel_parts = parts[i:]
            if rel_parts[-1].endswith(".lean"):
                rel_parts = rel_parts[:-1] + (rel_parts[-1][:-5],)
            return ".".join(rel_parts)

    # Try relative to base_path if provided
    if base_path:
        try:
            rel = file_path.relative_to(base_path)
            rel_parts = rel.parts
            if rel_parts[-1].endswith(".lean"):
                rel_parts = rel_parts[:-1] + (rel_parts[-1][:-5],)
            # Skip "src" directory
            if rel_parts and rel_parts[0] == "src":
                rel_parts = rel_parts[1:]
            return ".".join(rel_parts)
        except ValueError:
            pass

    # Fallback: just use stem
    return file_path.stem


def _extract_declarations_from_file(
    file_path: Path, module_prefix: str = "", base_path: Path | None = None
) -> list[LeanDeclaration]:
    """Extract declarations from a single Lean file.

    Uses regex patterns optimized for common Lean 4 declaration styles.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug(f"Could not read {file_path}: {e}")
        return []

    declarations: list[LeanDeclaration] = []

    # Determine module name
    module_name = module_prefix or _infer_module_name(file_path, base_path)

    # Track current namespace for qualified names
    namespace_stack: list[str] = []

    # Find namespace declarations
    namespace_pattern = re.compile(r"^namespace\s+([\w\.]+)", re.MULTILINE)
    end_pattern = re.compile(r"^end\s+([\w\.]+)?", re.MULTILINE)

    # Build namespace context map (line -> active namespace)
    lines = content.split("\n")
    namespace_at_line: dict[int, str] = {}
    current_ns = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if ns_match := namespace_pattern.match(stripped):
            namespace_stack.append(ns_match.group(1))
            current_ns = ".".join(namespace_stack)
        elif end_pattern.match(stripped):
            if namespace_stack:
                namespace_stack.pop()
                current_ns = ".".join(namespace_stack)
        namespace_at_line[i] = current_ns

    # Pattern to capture declarations with docstrings and signatures
    # Handles multi-line signatures by matching until := or where or |
    decl_pattern = re.compile(
        r"(?P<docstring>/--[\s\S]*?-/\s*)?"
        r"(?:@\[[\w\s,\(\)=\"\'\.]+\]\s*)*"  # attributes
        r"(?:private\s+|protected\s+|scoped\s+)?"
        r"(?:noncomputable\s+|unsafe\s+|partial\s+|nonrec\s+)*"
        r"(?P<kind>theorem|lemma|def|abbrev|class|structure|inductive|instance|axiom|opaque)\s+"
        r"(?P<name>[\w']+)"  # Just the base name, not qualified
        r"(?P<params>(?:\s*[\[\(\{][\s\S]*?[\]\)\}])*)"  # Parameters
        r"(?:\s*:\s*(?P<type>[^:=\n]+?))?"  # Optional type annotation
        r"(?=\s*(?::=|where|:|\||$))",  # Look ahead for definition start
        re.MULTILINE,
    )

    for match in decl_pattern.finditer(content):
        kind = match.group("kind")
        base_name = match.group("name")
        docstring = match.group("docstring")
        params = match.group("params") or ""
        type_ann = match.group("type") or ""

        # Clean up docstring
        if docstring:
            docstring = docstring.strip()
            if docstring.startswith("/--"):
                docstring = docstring[3:]
            if docstring.endswith("-/"):
                docstring = docstring[:-2]
            docstring = " ".join(docstring.split())  # Normalize whitespace

        # Build signature from params and type
        sig_parts = []
        if params:
            sig_parts.append(params.strip())
        if type_ann:
            sig_parts.append(f": {type_ann.strip()}")
        signature = " ".join(sig_parts)

        # Calculate line number
        line_num = content[: match.start()].count("\n") + 1

        # Skip private/internal names
        if base_name.startswith("_"):
            continue

        # Get namespace at this line
        ns = namespace_at_line.get(line_num - 1, "")

        # Build fully qualified name
        name_parts = [module_name]
        if ns:
            name_parts.append(ns)
        name_parts.append(base_name)
        full_name = ".".join(filter(None, name_parts))

        declarations.append(
            LeanDeclaration(
                name=full_name,
                kind=kind,
                module=module_name,
                signature=signature[:500] if signature else "",
                docstring=docstring[:1000] if docstring else None,
                file_path=str(file_path),
                line=line_num,
            )
        )

    return declarations


def _find_lean_files(
    root: Path, exclude_build: bool = True, max_files: int | None = None
) -> list[Path]:
    """Find all .lean files under root, excluding build directories."""
    files = []
    exclude_patterns = {
        ".lake/build",
        ".lake/packages/.lake",
        "__pycache__",
        ".git",
        "lake-packages",  # Old lake format
    }

    try:
        for lean_file in root.rglob("*.lean"):
            path_str = str(lean_file)
            if exclude_build and any(ex in path_str for ex in exclude_patterns):
                continue
            files.append(lean_file)
            if max_files and len(files) >= max_files:
                break
    except PermissionError:
        logger.warning(f"Permission denied accessing {root}")

    return files


def _get_embedding_function(
    provider: str = "default", model: str | None = None
) -> Any:
    """Get the appropriate embedding function based on provider.

    Args:
        provider: One of 'default', 'openai', 'gemini', 'voyage'
        model: Model name override

    Providers and default models:
        - default: all-MiniLM-L6-v2 (384 dims, local, fast, ~18KB/decl)
        - openai: text-embedding-3-large (3072 dims, best quality, ~50KB/decl)
        - gemini: text-embedding-004 (768 dims, good quality, ~25KB/decl)
        - voyage: voyage-code-2 (1024 dims, code-optimized, ~35KB/decl)

    Returns:
        A ChromaDB-compatible embedding function
    """
    try:
        import chromadb.utils.embedding_functions as ef
    except ImportError:
        raise RuntimeError(
            "chromadb is required for local leansearch. Install with: pip install chromadb"
        )

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set for OpenAI embeddings")
        # Use large model by default for best quality
        model_name = model or "text-embedding-3-large"
        logger.info(f"Using OpenAI embeddings: {model_name}")
        return ef.OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name)

    elif provider == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY or GEMINI_API_KEY not set for Gemini embeddings"
            )
        model_name = model or "models/text-embedding-004"
        logger.info(f"Using Gemini embeddings: {model_name}")
        return ef.GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key, model_name=model_name
        )

    elif provider == "voyage":
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set for Voyage embeddings")
        model_name = model or "voyage-code-2"
        logger.info(f"Using Voyage embeddings: {model_name}")
        return ef.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
            api_base="https://api.voyageai.com/v1",
        )

    else:
        # Default: use sentence-transformers (local, no API key)
        model_name = model or "all-MiniLM-L6-v2"
        logger.info(f"Using local sentence-transformers: {model_name}")
        return ef.SentenceTransformerEmbeddingFunction(model_name=model_name)


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
    ):
        self.project_root = project_root
        self.cache_dir = cache_dir or get_cache_dir()
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model

        self._client = None
        self._collection = None
        self._ready = False
        self._lock = asyncio.Lock()
        self._stats: IndexStats | None = None
        self._file_hashes: dict[str, str] = {}  # path -> hash for incremental indexing

    def _get_hash_file_path(self) -> Path:
        """Get path to file hash cache."""
        return self.cache_dir / "chroma" / f"{self._get_collection_name()}_hashes.json"

    def _load_file_hashes(self) -> dict[str, str]:
        """Load cached file hashes from disk."""
        hash_file = self._get_hash_file_path()
        if hash_file.exists():
            try:
                return json.loads(hash_file.read_text())
            except Exception:
                pass
        return {}

    def _save_file_hashes(self, hashes: dict[str, str]) -> None:
        """Save file hashes to disk."""
        hash_file = self._get_hash_file_path()
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        hash_file.write_text(json.dumps(hashes))

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def stats(self) -> IndexStats | None:
        return self._stats

    def _get_chroma_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise RuntimeError(
                    "chromadb is required for local leansearch. "
                    "Install with: pip install chromadb"
                )

            persist_dir = self.cache_dir / "chroma"
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))

        return self._client

    def _get_collection_name(self) -> str:
        """Get collection name based on project."""
        if self.project_root:
            project_hash = _compute_project_hash(self.project_root)
            return f"leansearch_{self.project_root.name}_{project_hash}"
        return "leansearch_global"

    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection for this project."""
        if self._collection is not None:
            return self._collection

        client = self._get_chroma_client()
        collection_name = self._get_collection_name()

        try:
            ef = _get_embedding_function(self.embedding_provider, self.embedding_model)
            self._collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

        return self._collection

    def _collect_lean_files(
        self, progress_callback: Callable[[str, int, int], None] | None = None
    ) -> list[tuple[Path, str, Path]]:
        """Collect all Lean files to index with their module prefixes and base paths.

        Returns list of (file_path, module_prefix, base_path) tuples.
        """
        files: list[tuple[Path, str, Path]] = []

        if not self.project_root:
            return files

        if progress_callback:
            progress_callback("Scanning project files...", 0, 1)

        # Project source files
        for src_dir in [".", "src", self.project_root.name]:
            src_path = self.project_root / src_dir
            if src_path.exists() and src_path.is_dir():
                lean_files = _find_lean_files(src_path)
                for f in lean_files:
                    files.append((f, "", src_path))
                if lean_files:
                    logger.info(f"Found {len(lean_files)} files in {src_path}")

        # .lake/packages dependencies
        lake_packages = self.project_root / ".lake" / "packages"
        if lake_packages.exists():
            pkg_dirs = [d for d in lake_packages.iterdir() if d.is_dir()]
            for idx, pkg_dir in enumerate(pkg_dirs):
                if progress_callback:
                    progress_callback(
                        f"Scanning {pkg_dir.name}...", idx, len(pkg_dirs)
                    )

                # Look for source files in common locations
                for subdir in [".", "src", pkg_dir.name]:
                    sub_path = pkg_dir / subdir
                    if sub_path.exists() and sub_path.is_dir():
                        pkg_files = _find_lean_files(sub_path)
                        for f in pkg_files:
                            files.append((f, "", sub_path))

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
        collection = self._get_or_create_collection()

        # Load cached file hashes for incremental indexing
        cached_hashes = {} if force else self._load_file_hashes()

        # Check if already indexed (and not forcing)
        if not force and collection.count() > 0 and cached_hashes:
            logger.info(
                f"Collection already has {collection.count()} items, skipping index"
            )
            self._ready = True
            return collection.count()

        # Clear existing data if forcing full reindex
        if force and collection.count() > 0:
            logger.info("Forcing reindex, clearing existing data...")
            try:
                client = self._get_chroma_client()
                client.delete_collection(self._get_collection_name())
                self._collection = None
                collection = self._get_or_create_collection()
                cached_hashes = {}
            except Exception as e:
                logger.warning(f"Could not clear collection: {e}")

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
            current_hash = _compute_file_hash(file_path)
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

        if not files_to_process and collection.count() > 0:
            logger.info("No files changed, index is up to date")
            self._ready = True
            self._save_file_hashes(new_hashes)
            return collection.count()

        if progress_callback:
            progress_callback(
                f"Extracting from {len(files_to_process)} files...", 0, len(files_to_process)
            )

        # Parallel extraction using ThreadPoolExecutor
        all_declarations: list[LeanDeclaration] = []
        kind_counts: dict[str, int] = {}
        max_workers = min(os.cpu_count() or 4, 8)  # Cap at 8 workers

        def extract_file(args: tuple[Path, str, Path]) -> list[LeanDeclaration]:
            file_path, module, base_path = args
            return _extract_declarations_from_file(file_path, module, base_path)

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
            self._save_file_hashes(new_hashes)
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

        # Prepare documents for ChromaDB
        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for i, decl in enumerate(all_declarations):
            # Create searchable document text with docstring weighting
            # Docstrings are repeated 3x to boost their importance for NL queries
            doc_parts = [decl.name.replace(".", " "), decl.kind]
            if decl.docstring:
                # Weight docstrings heavily - they contain natural language
                doc_parts.extend([decl.docstring] * 3)
            if decl.signature:
                doc_parts.append(decl.signature)
            doc = " ".join(doc_parts)

            documents.append(doc)
            metadatas.append(
                {
                    "name": decl.name,
                    "kind": decl.kind,
                    "module": decl.module,
                    "signature": decl.signature[:500],
                    "file_path": decl.file_path,
                    "line": decl.line,
                }
            )
            ids.append(f"decl_{i}_{hashlib.md5(decl.name.encode()).hexdigest()[:8]}")

        # Add to collection in batches
        batch_size = 500
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            i = batch_idx * batch_size
            batch_end = min(i + batch_size, len(documents))

            try:
                collection.add(
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    ids=ids[i:batch_end],
                )
            except Exception as e:
                logger.error(f"Failed to add batch {batch_idx}: {e}")

            if progress_callback:
                progress_callback(
                    f"Indexed {batch_end}/{len(documents)}...",
                    batch_end,
                    len(documents),
                )
            elif batch_idx % 10 == 0:
                logger.info(f"Indexed {batch_end}/{len(documents)} declarations")

        elapsed = time.monotonic() - start_time
        self._ready = True

        # Save file hashes for future incremental indexing
        self._save_file_hashes(new_hashes)

        self._stats = IndexStats(
            total_declarations=collection.count(),
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
            f"Indexing complete: {collection.count()} declarations in {elapsed:.1f}s "
            f"({files_added} new, {files_updated} updated, {files_unchanged} unchanged files)"
        )
        return collection.count()

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

        collection = self._get_or_create_collection()

        # Fetch more results for hybrid reranking
        fetch_count = min(num_results * 3, 50)

        try:
            results = collection.query(
                query_texts=[query],
                n_results=fetch_count,
                include=["metadatas", "distances", "documents"],
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        if not results or not results["metadatas"] or not results["metadatas"][0]:
            return []

        distances = results.get("distances", [[]])
        documents = results.get("documents", [[]])
        if not distances or not distances[0]:
            distances = [[0.0] * len(results["metadatas"][0])]
        if not documents or not documents[0]:
            documents = [[""] * len(results["metadatas"][0])]

        # Compute hybrid scores and deduplicate
        seen_names: set[str] = set()
        scored_results: list[tuple[float, dict[str, Any]]] = []

        for meta, dist, doc in zip(
            results["metadatas"][0], distances[0], documents[0]
        ):
            name = meta.get("name", "")

            # Deduplicate by name
            if name in seen_names:
                continue
            seen_names.add(name)

            # Hybrid scoring: combine semantic distance with keyword overlap
            # Lower distance = better semantic match
            # Higher keyword score = better keyword match
            semantic_score = 1.0 - min(dist, 1.0)  # Normalize to 0-1
            kw_score = _keyword_score(query, doc or name)

            # Weighted combination
            hybrid_score = (1 - hybrid_weight) * semantic_score + hybrid_weight * kw_score

            scored_results.append(
                (
                    hybrid_score,
                    {
                        "name": name,
                        "module": meta.get("module", ""),
                        "kind": meta.get("kind"),
                        "signature": meta.get("signature"),
                        "distance": dist,
                        "file_path": meta.get("file_path"),
                        "line": meta.get("line"),
                    },
                )
            )

        # Sort by hybrid score (descending) and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_results[:num_results]]

    async def ensure_indexed(self, project_root: Path | None = None) -> bool:
        """Ensure the project is indexed, indexing if necessary.

        This is the main entry point for lazy initialization.
        Thread-safe via asyncio lock.
        """
        async with self._lock:
            if project_root and project_root != self.project_root:
                # Project changed, need to reindex
                self.project_root = project_root
                self._collection = None
                self._ready = False

            if self._ready:
                return True

            try:
                count = self.index_project()
                return count >= 0
            except Exception as e:
                logger.error(f"Indexing failed: {e}")
                return False

    def reindex(
        self, progress_callback: Callable[[str, int, int], None] | None = None
    ) -> int:
        """Force reindex the project."""
        return self.index_project(force=True, progress_callback=progress_callback)

    def clear_cache(self) -> None:
        """Clear the cached index."""
        if self._collection is not None:
            try:
                client = self._get_chroma_client()
                client.delete_collection(self._get_collection_name())
            except Exception:
                pass
            self._collection = None
            self._ready = False
            self._stats = None


def check_leansearch_available() -> tuple[bool, str]:
    """Check if local leansearch dependencies are available."""
    import importlib.util

    if importlib.util.find_spec("chromadb") is not None:
        return True, ""
    return False, (
        "chromadb is required for local semantic search. "
        "Install with: pip install chromadb\n"
        "For better embeddings, also set one of:\n"
        "  - OPENAI_API_KEY (uses text-embedding-3-small)\n"
        "  - VOYAGE_API_KEY (uses voyage-code-2, excellent for code)"
    )

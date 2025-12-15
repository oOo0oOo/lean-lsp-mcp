"""Local semantic search for Lean 4 projects using embeddings.

This module provides local semantic search functionality similar to leansearch.net
but runs entirely locally with no rate limits. It indexes Lean declarations from
your project and dependencies, then uses vector embeddings for semantic search.

Supports multiple embedding backends:
- sentence-transformers (default, no API key required)
- OpenAI text-embedding-3-small/large
- Voyage AI voyage-code-2 (excellent for code)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    return hasher.hexdigest()[:16]


def _extract_declarations_from_file(
    file_path: Path, module_prefix: str = ""
) -> list[LeanDeclaration]:
    """Extract declarations from a single Lean file.

    This is a simplified extractor that uses regex patterns.
    For more complete extraction, jixia could be used.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug(f"Could not read {file_path}: {e}")
        return []

    declarations: list[LeanDeclaration] = []

    # Compute module name from file path
    module_name = module_prefix
    if not module_name:
        # Try to infer from file path relative to common patterns
        parts = file_path.parts
        for i, part in enumerate(parts):
            if part in ("src", "Mathlib", "Std", "Init", "Lean"):
                rel_parts = parts[i:]
                module_name = ".".join(rel_parts)[:-5] if rel_parts else ""  # strip .lean
                break
        if not module_name:
            module_name = file_path.stem

    # Pattern to capture declaration header with optional docstring and signature
    decl_pattern = re.compile(
        r"^(?P<docstring>/--[\s\S]*?-/\s*)?"
        r"(?:@\[[\w\s,]+\]\s*)*"
        r"(?:private\s+|protected\s+|scoped\s+)?"
        r"(?:noncomputable\s+|unsafe\s+|partial\s+|nonrec\s+)*"
        r"(?P<kind>theorem|lemma|def|abbrev|class|structure|inductive|instance|axiom|opaque)\s+"
        r"(?P<name>[\w'\.]+)"
        r"(?:\s*(?P<sig>[^:=]*:[^:=\n]+))?",
        re.MULTILINE,
    )

    for match in decl_pattern.finditer(content):
        kind = match.group("kind")
        name = match.group("name")
        docstring = match.group("docstring")
        sig = match.group("sig") or ""

        # Clean up docstring
        if docstring:
            docstring = docstring.strip()
            if docstring.startswith("/--"):
                docstring = docstring[3:]
            if docstring.endswith("-/"):
                docstring = docstring[:-2]
            docstring = docstring.strip()

        # Clean up signature
        sig = sig.strip()
        if sig.startswith(":"):
            sig = sig[1:].strip()

        # Calculate line number
        line_num = content[: match.start()].count("\n") + 1

        # Skip private/internal names
        if name.startswith("_"):
            continue

        full_name = f"{module_name}.{name}" if module_name else name

        declarations.append(
            LeanDeclaration(
                name=full_name,
                kind=kind,
                module=module_name,
                signature=sig[:500] if sig else "",  # Truncate long signatures
                docstring=docstring[:1000] if docstring else None,
                file_path=str(file_path),
                line=line_num,
            )
        )

    return declarations


def _find_lean_files(root: Path, exclude_build: bool = True) -> list[Path]:
    """Find all .lean files under root, excluding build directories."""
    files = []
    exclude_patterns = {".lake/build", ".lake/packages/.lake", "__pycache__"}

    for lean_file in root.rglob("*.lean"):
        path_str = str(lean_file)
        if exclude_build and any(ex in path_str for ex in exclude_patterns):
            continue
        files.append(lean_file)

    return files


def _get_embedding_function(
    provider: str = "default", model: str | None = None
) -> Any:
    """Get the appropriate embedding function based on provider.

    Args:
        provider: One of 'default', 'openai', 'voyage', 'anthropic'
        model: Model name override

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
        model_name = model or "text-embedding-3-small"
        return ef.OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name)

    elif provider == "voyage":
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set for Voyage embeddings")
        # Voyage is OpenAI-compatible
        model_name = model or "voyage-code-2"
        return ef.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
            api_base="https://api.voyageai.com/v1",
        )

    else:
        # Default: use sentence-transformers (local, no API key)
        model_name = model or "all-MiniLM-L6-v2"
        return ef.SentenceTransformerEmbeddingFunction(model_name=model_name)


class LeanSearchManager:
    """Manages local semantic search for Lean 4 projects.

    Features:
    - Indexes project source and .lake dependencies
    - Uses vector embeddings for semantic search
    - Supports multiple embedding backends (local or API-based)
    - Caches indices per project version
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

    @property
    def is_ready(self) -> bool:
        return self._ready

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

    def _collect_lean_files(self) -> list[tuple[Path, str]]:
        """Collect all Lean files to index with their module prefixes."""
        files: list[tuple[Path, str]] = []

        if not self.project_root:
            return files

        # Project source files
        for src_dir in [".", "src", self.project_root.name]:
            src_path = self.project_root / src_dir
            if src_path.exists():
                for f in _find_lean_files(src_path):
                    # Compute module prefix from relative path
                    try:
                        rel = f.relative_to(src_path)
                        module = ".".join(rel.parts[:-1] + (rel.stem,))
                    except ValueError:
                        module = f.stem
                    files.append((f, module))

        # .lake/packages dependencies
        lake_packages = self.project_root / ".lake" / "packages"
        if lake_packages.exists():
            for pkg_dir in lake_packages.iterdir():
                if not pkg_dir.is_dir():
                    continue
                # Look for source files in common locations
                for subdir in [".", "src", pkg_dir.name]:
                    sub_path = pkg_dir / subdir
                    if sub_path.exists():
                        for f in _find_lean_files(sub_path):
                            try:
                                rel = f.relative_to(sub_path)
                                module = ".".join(rel.parts[:-1] + (rel.stem,))
                            except ValueError:
                                module = f.stem
                            files.append((f, module))

        return files

    def index_project(self, force: bool = False) -> int:
        """Index the project and dependencies.

        Args:
            force: If True, reindex even if cache exists

        Returns:
            Number of declarations indexed
        """
        collection = self._get_or_create_collection()

        # Check if already indexed (and not forcing)
        if not force and collection.count() > 0:
            logger.info(
                f"Collection already has {collection.count()} items, skipping index"
            )
            self._ready = True
            return collection.count()

        logger.info("Starting project indexing...")
        files = self._collect_lean_files()
        logger.info(f"Found {len(files)} Lean files to index")

        all_declarations: list[LeanDeclaration] = []
        for file_path, module in files:
            decls = _extract_declarations_from_file(file_path, module)
            all_declarations.extend(decls)

        if not all_declarations:
            logger.warning("No declarations found to index")
            self._ready = True
            return 0

        logger.info(f"Extracted {len(all_declarations)} declarations")

        # Prepare documents for ChromaDB
        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for i, decl in enumerate(all_declarations):
            # Create searchable document text
            doc_parts = [decl.name, decl.kind]
            if decl.signature:
                doc_parts.append(decl.signature)
            if decl.docstring:
                doc_parts.append(decl.docstring)
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
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            try:
                collection.add(
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    ids=ids[i:batch_end],
                )
            except Exception as e:
                logger.error(f"Failed to add batch {i}-{batch_end}: {e}")
                # Continue with other batches

            if (i // batch_size) % 10 == 0:
                logger.info(f"Indexed {batch_end}/{len(documents)} declarations")

        self._ready = True
        logger.info(f"Indexing complete: {collection.count()} declarations")
        return collection.count()

    def search(
        self, query: str, num_results: int = 5
    ) -> list[dict[str, Any]]:
        """Search for declarations matching a natural language query.

        Args:
            query: Natural language search query
            num_results: Maximum number of results

        Returns:
            List of matching declarations with metadata
        """
        if not self._ready:
            raise RuntimeError("Index not ready. Call index_project() first.")

        collection = self._get_or_create_collection()

        try:
            results = collection.query(
                query_texts=[query],
                n_results=num_results,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        if not results or not results["metadatas"] or not results["metadatas"][0]:
            return []

        output = []
        for meta, dist in zip(
            results["metadatas"][0],
            results["distances"][0] if results["distances"] else [0] * len(results["metadatas"][0]),
        ):
            output.append(
                {
                    "name": meta.get("name", ""),
                    "module": meta.get("module", ""),
                    "kind": meta.get("kind"),
                    "signature": meta.get("signature"),
                    "distance": dist,
                }
            )

        return output

    async def ensure_indexed(self, project_root: Path | None = None) -> bool:
        """Ensure the project is indexed, indexing if necessary.

        This is the main entry point for lazy initialization.
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

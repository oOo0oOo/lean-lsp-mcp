"""Jixia integration for extracting Lean 4 declarations and references.

Jixia is a static analysis tool for Lean 4 that extracts:
- Declarations: name, kind, signature, docstring
- Symbols: type references and value references (for SineQuaNon triggers)

This module provides a wrapper to run jixia and parse its output.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .models import LeanDeclaration

logger = logging.getLogger(__name__)


def get_jixia_path() -> Path | None:
    """Find the jixia binary.

    Checks in order:
    1. JIXIA_PATH environment variable
    2. ~/.cache/lean-lsp-mcp/jixia/.lake/build/bin/jixia
    3. jixia on PATH
    """
    if env_path := os.environ.get("JIXIA_PATH"):
        path = Path(env_path)
        if path.exists():
            return path

    cache_path = Path.home() / ".cache/lean-lsp-mcp/jixia/.lake/build/bin/jixia"
    if cache_path.exists():
        return cache_path

    if which := shutil.which("jixia"):
        return Path(which)

    return None


def check_jixia_available() -> tuple[bool, str]:
    """Check if jixia is available and return status + message."""
    path = get_jixia_path()
    if path is None:
        return False, (
            "jixia not found. Install with:\n"
            "  cd ~/.cache/lean-lsp-mcp\n"
            "  git clone https://github.com/frenzymath/jixia\n"
            "  cd jixia && lake build"
        )
    return True, f"jixia found at {path}"


@dataclass
class JixiaDeclaration:
    """A declaration extracted by jixia."""

    name: str  # Fully qualified name like "List.map_length"
    kind: str  # theorem, def, inductive, etc.
    module: str  # Module path
    signature: str  # Pretty-printed type signature
    docstring: str | None
    file_path: str
    line: int
    type_references: list[str] = field(default_factory=list)  # Constants in type
    value_references: list[str] = field(default_factory=list)  # Constants in value


@dataclass
class JixiaResult:
    """Result of jixia extraction."""

    declarations: list[JixiaDeclaration]
    references: dict[str, set[str]]  # decl_name -> set of constants it uses


class JixiaExtractor:
    """Extract declarations and references using jixia."""

    def __init__(self, jixia_path: Path | None = None):
        self.jixia_path = jixia_path or get_jixia_path()

    def is_available(self) -> bool:
        """Check if jixia is available."""
        return self.jixia_path is not None and self.jixia_path.exists()

    def extract_file(
        self,
        lean_file: Path,
        project_root: Path | None = None,
    ) -> JixiaResult:
        """Extract declarations and references from a single Lean file.

        Args:
            lean_file: Path to the .lean file
            project_root: Project root for running with lake env

        Returns:
            JixiaResult with declarations and reference graph
        """
        if not self.is_available():
            raise RuntimeError("jixia not available")

        # Create temp files for output
        with (
            tempfile.NamedTemporaryFile(suffix=".json", delete=False) as decl_file,
            tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sym_file,
        ):
            decl_path = decl_file.name
            sym_path = sym_file.name

        try:
            # Build command
            cmd = [
                str(self.jixia_path),
                "-d",
                decl_path,
                "-s",
                sym_path,
                str(lean_file),
            ]

            # Run with lake env if in a project
            if project_root:
                cmd = ["lake", "env"] + cmd

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(project_root) if project_root else None,
                timeout=60,
            )

            if result.returncode != 0:
                # Non-fatal - file might just have errors
                logger.debug(f"jixia warning for {lean_file}: {result.stderr[:200]}")

            # Parse outputs
            declarations = []
            references: dict[str, set[str]] = {}

            # Parse declarations
            decl_data = []
            if Path(decl_path).exists():
                try:
                    decl_data = json.loads(Path(decl_path).read_text())
                except json.JSONDecodeError:
                    pass

            # Parse symbols for references
            sym_data = []
            if Path(sym_path).exists():
                try:
                    sym_data = json.loads(Path(sym_path).read_text())
                except json.JSONDecodeError:
                    pass

            # Build symbol lookup for references
            sym_by_name: dict[str, dict] = {}
            for sym in sym_data:
                name = ".".join(sym.get("name", []))
                sym_by_name[name] = sym

            # Process declarations
            for decl in decl_data:
                name_parts = decl.get("name", [])
                if not name_parts:
                    continue

                name = ".".join(name_parts)
                kind = decl.get("kind", "unknown")

                # Skip internal/generated declarations
                if kind in ("constructor", "recursor", "example"):
                    continue
                if "_sizeOf_" in name or "_unsafe_rec" in name:
                    continue
                if name.startswith("_"):
                    continue

                # Get signature from declaration
                sig_info = decl.get("signature", {})
                signature = sig_info.get("pp", "") if isinstance(sig_info, dict) else ""

                # Get docstring
                modifiers = decl.get("modifiers", {})
                doc_info = modifiers.get("docString")
                docstring = (
                    doc_info[0] if doc_info and isinstance(doc_info, list) else None
                )

                # Get line number from range
                id_info = decl.get("id", {})
                range_info = (
                    id_info.get("range", [0, 0])
                    if isinstance(id_info, dict)
                    else [0, 0]
                )
                line = range_info[0] if range_info else 0

                # Get references from symbol data
                type_refs = []
                value_refs = []
                if name in sym_by_name:
                    sym = sym_by_name[name]
                    type_refs = [
                        ".".join(r) for r in sym.get("typeReferences", []) if r
                    ]
                    value_refs = [
                        ".".join(r)
                        for r in sym.get("valueReferences", [])
                        if r and r[0] not in ("Eq", "rfl")
                    ]

                # Create declaration
                jd = JixiaDeclaration(
                    name=name,
                    kind=kind,
                    module=name_parts[0] if len(name_parts) > 1 else "",
                    signature=signature[:500] if signature else "",
                    docstring=docstring[:1000] if docstring else None,
                    file_path=str(lean_file),
                    line=line,
                    type_references=type_refs,
                    value_references=value_refs,
                )
                declarations.append(jd)

                # Build reference graph (combine type and value refs)
                all_refs = set(type_refs) | set(value_refs)
                if all_refs:
                    references[name] = all_refs

            return JixiaResult(declarations=declarations, references=references)

        finally:
            # Clean up temp files
            Path(decl_path).unlink(missing_ok=True)
            Path(sym_path).unlink(missing_ok=True)

    def extract_project(
        self,
        project_root: Path,
        include_deps: bool = True,
        max_files: int | None = None,
    ) -> JixiaResult:
        """Extract declarations from an entire project.

        Args:
            project_root: Root of the Lean project
            include_deps: Whether to include .lake/packages dependencies
            max_files: Maximum number of files to process

        Returns:
            Combined JixiaResult for all files
        """
        from .declarations import find_lean_files

        all_decls: list[JixiaDeclaration] = []
        all_refs: dict[str, set[str]] = {}

        # Find all lean files
        files = find_lean_files(project_root, exclude_build=True, max_files=max_files)

        # Include dependencies if requested
        if include_deps:
            lake_packages = project_root / ".lake" / "packages"
            if lake_packages.exists():
                for pkg_dir in lake_packages.iterdir():
                    if pkg_dir.is_dir() and not pkg_dir.name.startswith("."):
                        pkg_files = find_lean_files(pkg_dir, exclude_build=True)
                        files.extend(pkg_files)
                        if max_files and len(files) >= max_files:
                            files = files[:max_files]
                            break

        logger.info(f"Extracting {len(files)} files with jixia...")

        for i, lean_file in enumerate(files):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(files)} files")

            try:
                result = self.extract_file(lean_file, project_root)
                all_decls.extend(result.declarations)
                all_refs.update(result.references)
            except Exception as e:
                logger.debug(f"Failed to extract {lean_file}: {e}")

        return JixiaResult(declarations=all_decls, references=all_refs)


def jixia_to_lean_declarations(
    jixia_result: JixiaResult,
) -> list[LeanDeclaration]:
    """Convert JixiaDeclarations to LeanDeclarations for indexing."""
    return [
        LeanDeclaration(
            name=jd.name,
            kind=jd.kind,
            module=jd.module,
            signature=jd.signature,
            docstring=jd.docstring,
            file_path=jd.file_path,
            line=jd.line,
        )
        for jd in jixia_result.declarations
    ]

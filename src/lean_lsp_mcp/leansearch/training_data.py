"""Integration with lean-training-data for accurate declaration extraction.

The lean-training-data project provides metaprogramming tools that extract
declarations and dependencies directly from the Lean 4 environment, which
is more accurate than regex-based extraction.

Tools:
- declaration_types: Extracts kind, name, and type for all declarations
- premises: Extracts dependency graph between declarations

See: https://github.com/semorrison/lean-training-data
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .models import LeanDeclaration, PremiseEdge, PremiseGraph

logger = logging.getLogger(__name__)


class TrainingDataExtractor:
    """Extract declarations and premises using lean-training-data tools.

    This class wraps the lean-training-data executables to extract structured
    information from Lean 4 projects with full type information.
    """

    def __init__(self, repo_path: Path):
        """Initialize with path to lean-training-data repository.

        Args:
            repo_path: Path to cloned lean-training-data repo
        """
        self.repo_path = repo_path
        self._built = False

    def is_available(self) -> bool:
        """Check if lean-training-data is available."""
        return self.repo_path.exists() and (
            self.repo_path / "lakefile.lean"
        ).exists()

    def ensure_built(self) -> bool:
        """Build lean-training-data if binaries don't exist.

        Returns True if build succeeds or already built.
        """
        if self._built:
            return True

        bin_path = self.repo_path / ".lake/build/bin/declaration_types"
        if bin_path.exists():
            self._built = True
            return True

        if not self.is_available():
            logger.warning(f"lean-training-data not found at {self.repo_path}")
            return False

        logger.info("Building lean-training-data...")
        try:
            result = subprocess.run(
                ["lake", "build"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes max
            )
            if result.returncode != 0:
                logger.error(f"Build failed: {result.stderr}")
                return False
            self._built = True
            return True
        except subprocess.TimeoutExpired:
            logger.error("Build timed out after 10 minutes")
            return False
        except Exception as e:
            logger.error(f"Build error: {e}")
            return False

    def extract_declarations(
        self, module: str = "Mathlib", timeout: int = 300
    ) -> list[LeanDeclaration]:
        """Extract declarations using declaration_types tool.

        Args:
            module: Module to extract from (e.g., "Mathlib", "Std")
            timeout: Maximum time in seconds

        Returns:
            List of extracted declarations
        """
        if not self.ensure_built():
            return []

        bin_path = self.repo_path / ".lake/build/bin/declaration_types"
        if not bin_path.exists():
            logger.warning("declaration_types binary not found")
            return []

        try:
            logger.info(f"Extracting declarations from {module}...")
            result = subprocess.run(
                [str(bin_path), module],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                logger.warning(f"declaration_types failed: {result.stderr[:500]}")
                return []

            return self._parse_declaration_types(result.stdout)

        except subprocess.TimeoutExpired:
            logger.warning(f"declaration_types timed out after {timeout}s")
            return []
        except Exception as e:
            logger.error(f"Error running declaration_types: {e}")
            return []

    def _parse_declaration_types(self, output: str) -> list[LeanDeclaration]:
        """Parse declaration_types output.

        Format:
        ---
        theorem
        Nat.add_comm
        ∀ (n m : Nat), n + m = m + n
        ---
        def
        List.map
        {α β : Type u_1} → (α → β) → List α → List β
        """
        declarations = []

        for block in output.split("---\n"):
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n", 2)
            if len(lines) < 2:
                continue

            kind = lines[0].strip()
            name = lines[1].strip()
            signature = lines[2].strip() if len(lines) > 2 else ""

            # Skip internal names
            if name.startswith("_") or "._" in name:
                continue

            # Infer module from name
            parts = name.rsplit(".", 1)
            module = parts[0] if len(parts) > 1 else ""

            declarations.append(
                LeanDeclaration(
                    name=name,
                    kind=kind,
                    module=module,
                    signature=signature[:500],
                    docstring=None,  # declaration_types doesn't extract docs
                    file_path="",  # Not available from this tool
                    line=0,
                )
            )

        logger.info(f"Parsed {len(declarations)} declarations")
        return declarations

    def extract_premises(
        self, module: str = "Mathlib", timeout: int = 300
    ) -> PremiseGraph:
        """Extract premise graph using premises tool.

        Args:
            module: Module to extract from
            timeout: Maximum time in seconds

        Returns:
            PremiseGraph with dependency information
        """
        if not self.ensure_built():
            return PremiseGraph()

        bin_path = self.repo_path / ".lake/build/bin/premises"
        if not bin_path.exists():
            logger.warning("premises binary not found")
            return PremiseGraph()

        try:
            logger.info(f"Extracting premises from {module}...")
            result = subprocess.run(
                [str(bin_path), module],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                logger.warning(f"premises failed: {result.stderr[:500]}")
                return PremiseGraph()

            return self._parse_premises(result.stdout)

        except subprocess.TimeoutExpired:
            logger.warning(f"premises timed out after {timeout}s")
            return PremiseGraph()
        except Exception as e:
            logger.error(f"Error running premises: {e}")
            return PremiseGraph()

    def _parse_premises(self, output: str) -> PremiseGraph:
        """Parse premises output.

        Format:
        ---
        Nat.add_comm
        * Nat.add        (explicit premise, marked with *)
        s Nat.succ_add   (simp premise, marked with s)
          Nat.zero_add   (implicit premise, just spaces)
        """
        graph = PremiseGraph()

        for block in output.split("---\n"):
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")
            if not lines:
                continue

            name = lines[0].strip()
            if not name or name.startswith("_"):
                continue

            edges = []
            for line in lines[1:]:
                if not line or len(line) < 2:
                    continue

                prefix = line[:2]
                target = line[2:].strip()

                if not target or target.startswith("_"):
                    continue

                if prefix == "* ":
                    edges.append(PremiseEdge(target, is_explicit=True, is_simp=False))
                elif prefix == "s ":
                    edges.append(PremiseEdge(target, is_explicit=False, is_simp=True))
                elif prefix == "  ":
                    edges.append(PremiseEdge(target, is_explicit=False, is_simp=False))

            if edges:
                graph.adjacency[name] = edges

        graph.build_reverse_index()
        logger.info(
            f"Parsed premise graph with {len(graph.adjacency)} declarations"
        )
        return graph


def find_training_data_repo() -> Path | None:
    """Try to find lean-training-data repository.

    Searches in common locations relative to the lean-lsp-mcp installation.
    """
    # Common locations to check
    candidates = [
        Path.home() / "lean-training-data",
        Path.home() / "repos" / "lean-training-data",
        Path.home() / "src" / "lean-training-data",
        Path(__file__).parent.parent.parent.parent / "lean-training-data",
    ]

    for path in candidates:
        if path.exists() and (path / "lakefile.lean").exists():
            return path

    return None

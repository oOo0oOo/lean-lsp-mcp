"""Local Loogle installation and subprocess management.

This module provides automated installation and management of a local Loogle
instance, avoiding rate limits and network dependencies of the remote API.
"""

from __future__ import annotations

import json
import logging
import os
import select
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Get the cache directory, respecting XDG_CACHE_HOME."""
    if cache_dir := os.environ.get("LEAN_LOOGLE_CACHE_DIR"):
        return Path(cache_dir)
    xdg_cache = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(xdg_cache) / "lean-lsp-mcp" / "loogle"


class LoogleManager:
    """Manages local loogle installation and subprocess.

    Handles:
    - Cloning the loogle repository
    - Building loogle and downloading mathlib cache
    - Building and caching the search index
    - Running loogle as an interactive subprocess
    - Query communication via stdin/stdout
    """

    REPO_URL = "https://github.com/nomeata/loogle.git"
    READY_SIGNAL = "Loogle is ready.\n"

    def __init__(self, cache_dir: Path | None = None):
        """Initialize LoogleManager.

        Args:
            cache_dir: Override cache directory (default: ~/.cache/lean-lsp-mcp/loogle)
        """
        self.cache_dir = cache_dir or get_cache_dir()
        self.repo_dir = self.cache_dir / "repo"
        self.index_dir = self.cache_dir / "index"
        self.process: subprocess.Popen[bytes] | None = None
        self._ready = False
        self._restart_count = 0

    @property
    def binary_path(self) -> Path:
        """Path to the loogle binary."""
        return self.repo_dir / ".lake" / "build" / "bin" / "loogle"

    @property
    def is_installed(self) -> bool:
        """Check if loogle is installed and binary exists."""
        return self.binary_path.exists()

    def _check_prerequisites(self) -> tuple[bool, str]:
        """Check if required tools (git, lake/elan) are available.

        Returns:
            Tuple of (success, error_message)
        """
        # Check git
        if not shutil.which("git"):
            return False, "git not found in PATH"

        # Check lake (part of elan)
        if not shutil.which("lake"):
            return False, "lake not found in PATH (install elan: https://github.com/leanprover/elan)"

        return True, ""

    def _clone_repo(self) -> bool:
        """Clone loogle repository if not present.

        Returns:
            True if repo exists or was cloned successfully
        """
        if self.repo_dir.exists():
            logger.debug(f"Loogle repo already exists at {self.repo_dir}")
            return True

        logger.info(f"Cloning loogle to {self.repo_dir}...")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", self.REPO_URL, str(self.repo_dir)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout for clone
            )
            if result.returncode != 0:
                logger.error(f"Git clone failed: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("Git clone timed out")
            return False
        except Exception as e:
            logger.error(f"Git clone error: {e}")
            return False

    def _get_lake_env(self) -> dict[str, str]:
        """Get environment variables for lake commands.

        Disables LAKE_ARTIFACT_CACHE to ensure olean files are stored in
        traditional paths that loogle expects (not content-addressable storage).
        """
        env = os.environ.copy()
        env["LAKE_ARTIFACT_CACHE"] = "false"
        return env

    def _build_loogle(self) -> bool:
        """Build loogle binary.

        Runs:
        1. lake exe cache get (download mathlib cache)
        2. lake build (build loogle)

        Returns:
            True if build successful
        """
        if self.is_installed:
            logger.debug("Loogle binary already exists")
            return True

        if not self.repo_dir.exists():
            logger.error("Repo not cloned, cannot build")
            return False

        lake_env = self._get_lake_env()

        # Download mathlib cache first
        logger.info("Downloading mathlib cache (this may take a few minutes)...")
        try:
            result = subprocess.run(
                ["lake", "exe", "cache", "get"],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
                env=lake_env,
            )
            if result.returncode != 0:
                logger.warning(f"Cache get warning: {result.stderr}")
                # Continue anyway, build might still work
        except subprocess.TimeoutExpired:
            logger.warning("Cache download timed out, continuing with build...")
        except Exception as e:
            logger.warning(f"Cache download error: {e}, continuing...")

        # Build loogle
        logger.info("Building loogle (this may take several minutes)...")
        try:
            result = subprocess.run(
                ["lake", "build"],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=900,  # 15 min timeout for build
                env=lake_env,
            )
            if result.returncode != 0:
                logger.error(f"Build failed: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("Build timed out")
            return False
        except Exception as e:
            logger.error(f"Build error: {e}")
            return False

    def _get_mathlib_version(self) -> str:
        """Get mathlib version from lake-manifest.json."""
        manifest_path = self.repo_dir / "lake-manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                for pkg in manifest.get("packages", []):
                    if pkg.get("name") == "mathlib":
                        return pkg.get("rev", "unknown")[:12]
            except Exception:
                pass
        return "unknown"

    def _get_index_path(self) -> Path:
        """Get path for the index file based on mathlib version."""
        version = self._get_mathlib_version()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        return self.index_dir / f"mathlib-{version}.idx"

    def _build_index(self) -> Path | None:
        """Build and cache the search index.

        Returns:
            Path to index file if successful, None otherwise
        """
        index_path = self._get_index_path()

        if index_path.exists():
            logger.debug(f"Index already exists at {index_path}")
            return index_path

        if not self.is_installed:
            logger.error("Loogle not installed, cannot build index")
            return None

        logger.info(f"Building search index (this may take 1-5 minutes)...")

        try:
            # Run loogle with --write-index to build and save index
            # Use an empty query to just build the index
            result = subprocess.run(
                [
                    str(self.binary_path),
                    "--write-index",
                    str(index_path),
                    "--json",
                    "",  # Empty query
                ],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
            )
            # Even if the query fails, the index should be written
            if index_path.exists():
                logger.info(f"Index built successfully at {index_path}")
                return index_path
            else:
                logger.error(f"Index not created. Stdout: {result.stdout[:500]}, Stderr: {result.stderr[:500]}")
                return None
        except subprocess.TimeoutExpired:
            logger.error("Index build timed out")
            return None
        except Exception as e:
            logger.error(f"Index build error: {e}")
            return None

    def ensure_installed(self) -> bool:
        """Ensure loogle is fully installed and ready.

        This is the main entry point for installation. It:
        1. Checks prerequisites (git, lake)
        2. Clones the repo if needed
        3. Builds loogle if needed
        4. Builds the index if needed

        Returns:
            True if loogle is ready to use
        """
        # Check prerequisites
        ok, error = self._check_prerequisites()
        if not ok:
            logger.warning(f"Prerequisites check failed: {error}")
            return False

        # Clone repo
        if not self._clone_repo():
            return False

        # Build loogle
        if not self._build_loogle():
            return False

        # Build index (optional but recommended)
        index_path = self._build_index()
        if not index_path:
            logger.warning("Index build failed, loogle will build index on startup (slower)")

        return self.is_installed

    def start(self) -> bool:
        """Start the loogle subprocess in interactive mode.

        Returns:
            True if subprocess started and is ready
        """
        if self.process is not None and self.process.poll() is None:
            logger.debug("Loogle process already running")
            return self._ready

        if not self.is_installed:
            logger.error("Loogle not installed")
            return False

        index_path = self._get_index_path()
        cmd = [str(self.binary_path), "--json", "--interactive"]

        # Use cached index if available
        if index_path.exists():
            cmd.extend(["--read-index", str(index_path)])
            logger.debug(f"Using cached index: {index_path}")

        logger.info("Starting loogle subprocess...")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.repo_dir,
            )

            # Wait for ready signal
            if self.process.stdout:
                # Use select for non-blocking read with timeout
                ready, _, _ = select.select([self.process.stdout], [], [], 120)  # 2 min timeout
                if ready:
                    line = self.process.stdout.readline().decode()
                    if self.READY_SIGNAL in line:
                        self._ready = True
                        logger.info("Loogle subprocess ready")
                        return True
                    else:
                        logger.error(f"Unexpected startup output: {line}")
                else:
                    logger.error("Timeout waiting for loogle to become ready")

            return False
        except Exception as e:
            logger.error(f"Failed to start loogle: {e}")
            return False

    def query(self, q: str, num_results: int = 8) -> list[dict[str, Any]]:
        """Send a query to loogle and return results.

        Args:
            q: The loogle query string
            num_results: Maximum number of results to return

        Returns:
            List of result dicts with 'name', 'type', 'module' keys

        Raises:
            RuntimeError: If loogle is not running or query fails
        """
        if not self._ready or self.process is None:
            # Try to restart once
            if self._restart_count < 1:
                self._restart_count += 1
                logger.info("Attempting to restart loogle subprocess...")
                if self.start():
                    return self.query(q, num_results)
            raise RuntimeError("Loogle subprocess not ready")

        if self.process.poll() is not None:
            # Process died, try restart
            self._ready = False
            if self._restart_count < 1:
                self._restart_count += 1
                logger.info("Loogle process died, restarting...")
                if self.start():
                    return self.query(q, num_results)
            raise RuntimeError("Loogle subprocess died and restart failed")

        try:
            # Send query
            assert self.process.stdin is not None
            assert self.process.stdout is not None

            self.process.stdin.write(f"{q}\n".encode())
            self.process.stdin.flush()

            # Read response (single JSON line)
            ready, _, _ = select.select([self.process.stdout], [], [], 30)  # 30s timeout
            if not ready:
                raise RuntimeError("Query timeout")

            line = self.process.stdout.readline().decode()
            if not line:
                raise RuntimeError("Empty response from loogle")

            response = json.loads(line)

            # Handle errors
            if error := response.get("error"):
                logger.warning(f"Loogle query error: {error}")
                return []

            # Extract hits
            hits = response.get("hits", [])
            return [
                {
                    "name": h.get("name", ""),
                    "type": h.get("type", ""),
                    "module": h.get("module", ""),
                }
                for h in hits[:num_results]
            ]

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise RuntimeError(f"Invalid loogle response: {e}") from e
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise

    def stop(self) -> None:
        """Stop the loogle subprocess."""
        if self.process is not None:
            logger.debug("Stopping loogle subprocess...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                logger.warning(f"Error stopping loogle: {e}")
            finally:
                self.process = None
                self._ready = False
                self._restart_count = 0

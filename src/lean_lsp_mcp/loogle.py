"""Loogle search - local subprocess and remote API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import orjson

from lean_lsp_mcp.models import LoogleResult

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    if d := os.environ.get("LEAN_LOOGLE_CACHE_DIR"):
        return Path(d)
    xdg = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(xdg) / "lean-lsp-mcp" / "loogle"


def loogle_remote(query: str, num_results: int) -> list[LoogleResult] | str:
    """Query the remote loogle API."""
    try:
        req = urllib.request.Request(
            f"https://loogle.lean-lang.org/json?q={urllib.parse.quote(query)}",
            headers={"User-Agent": "lean-lsp-mcp/0.1"},
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            results = orjson.loads(response.read())
        if "hits" not in results:
            return "No results found."
        hits = results["hits"][:num_results]
        return [
            LoogleResult(
                name=r.get("name", ""),
                type=r.get("type", ""),
                module=r.get("module", ""),
            )
            for r in hits
        ]
    except Exception as e:
        return f"loogle error:\n{e}"


class LoogleManager:
    """Manages local loogle installation and async subprocess."""

    REPO_URL = "https://github.com/nomeata/loogle.git"
    READY_SIGNAL = "Loogle is ready."

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or get_cache_dir()
        self.repo_dir = self.cache_dir / "repo"
        self.index_dir = self.cache_dir / "index"
        self.process: asyncio.subprocess.Process | None = None
        self._ready = False
        self._lock = asyncio.Lock()

    @property
    def binary_path(self) -> Path:
        return self.repo_dir / ".lake" / "build" / "bin" / "loogle"

    @property
    def is_installed(self) -> bool:
        return self.binary_path.exists()

    @property
    def is_running(self) -> bool:
        return (
            self._ready and self.process is not None and self.process.returncode is None
        )

    def _check_prerequisites(self) -> tuple[bool, str]:
        if not shutil.which("git"):
            return False, "git not found in PATH"
        if not shutil.which("lake"):
            return (
                False,
                "lake not found (install elan: https://github.com/leanprover/elan)",
            )
        return True, ""

    def _run(
        self, cmd: list[str], timeout: int = 300, cwd: Path | None = None
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["LAKE_ARTIFACT_CACHE"] = "false"
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or self.repo_dir,
            env=env,
        )

    def _clone_repo(self) -> bool:
        if self.repo_dir.exists():
            return True
        logger.info(f"Cloning loogle to {self.repo_dir}...")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            r = self._run(
                ["git", "clone", "--depth", "1", self.REPO_URL, str(self.repo_dir)],
                cwd=self.cache_dir,
            )
            if r.returncode != 0:
                logger.error(f"Clone failed: {r.stderr}")
                return False
            return True
        except Exception as e:
            logger.error(f"Clone error: {e}")
            return False

    def _build_loogle(self) -> bool:
        if self.is_installed:
            return True
        if not self.repo_dir.exists():
            return False
        logger.info("Downloading mathlib cache...")
        try:
            self._run(["lake", "exe", "cache", "get"], timeout=600)
        except Exception as e:
            logger.warning(f"Cache download: {e}")
        logger.info("Building loogle...")
        try:
            return self._run(["lake", "build"], timeout=900).returncode == 0
        except Exception as e:
            logger.error(f"Build error: {e}")
            return False

    def _get_mathlib_version(self) -> str:
        try:
            manifest = json.loads((self.repo_dir / "lake-manifest.json").read_text())
            for pkg in manifest.get("packages", []):
                if pkg.get("name") == "mathlib":
                    return pkg.get("rev", "unknown")[:12]
        except Exception:
            pass
        return "unknown"

    def _get_toolchain_version(self) -> str | None:
        """Get the Lean toolchain version from lean-toolchain file."""
        try:
            return (self.repo_dir / "lean-toolchain").read_text().strip()
        except Exception:
            return None

    def _check_toolchain_installed(self) -> tuple[bool, str]:
        """Check if the required Lean toolchain is installed."""
        tc = self._get_toolchain_version()
        if not tc:
            return True, ""  # Can't check without lean-toolchain file
        # Convert lean-toolchain format to elan directory name
        # e.g., "leanprover/lean4:v4.25.0-rc1" -> "leanprover--lean4---v4.25.0-rc1"
        tc_dir_name = tc.replace("/", "--").replace(":", "---")
        elan_home = Path(os.environ.get("ELAN_HOME", Path.home() / ".elan"))
        tc_path = elan_home / "toolchains" / tc_dir_name
        if not tc_path.exists():
            return False, (
                f"Toolchain '{tc}' not installed. "
                f"Run: cd {self.repo_dir} && lake update"
            )
        return True, ""

    def check_environment(self) -> tuple[bool, str]:
        """Check if the loogle environment is valid. Returns (ok, error_msg)."""
        if not self.is_installed:
            return False, "Loogle binary not found"
        ok, err = self._check_toolchain_installed()
        if not ok:
            return False, err
        return True, ""

    def _get_index_path(self) -> Path:
        return self.index_dir / f"mathlib-{self._get_mathlib_version()}.idx"

    def _cleanup_old_indices(self) -> None:
        """Remove old index files from previous mathlib versions."""
        if not self.index_dir.exists():
            return
        current = self._get_index_path()
        for idx in self.index_dir.glob("*.idx"):
            if idx != current:
                try:
                    idx.unlink()
                    logger.info(f"Removed old index: {idx.name}")
                except Exception:
                    pass

    def _build_index(self) -> Path | None:
        index_path = self._get_index_path()
        if index_path.exists():
            return index_path
        if not self.is_installed:
            return None
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_old_indices()
        logger.info("Building search index...")
        try:
            self._run(
                [str(self.binary_path), "--write-index", str(index_path), "--json", ""],
                timeout=600,
            )
            return index_path if index_path.exists() else None
        except Exception as e:
            logger.error(f"Index build error: {e}")
            return None

    def ensure_installed(self) -> bool:
        ok, err = self._check_prerequisites()
        if not ok:
            logger.warning(f"Prerequisites: {err}")
            return False
        if not self._clone_repo() or not self._build_loogle():
            return False
        if not self._build_index():
            logger.warning("Index build failed, loogle will build on startup")
        return self.is_installed

    async def start(self) -> bool:
        if self.process is not None and self.process.returncode is None:
            return self._ready
        ok, err = self.check_environment()
        if not ok:
            logger.error(f"Loogle environment check failed: {err}")
            return False
        cmd = [str(self.binary_path), "--json", "--interactive"]
        if (idx := self._get_index_path()).exists():
            cmd.extend(["--read-index", str(idx)])
        logger.info("Starting loogle subprocess...")
        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.repo_dir,
            )
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=120)
            decoded = line.decode()
            if self.READY_SIGNAL in decoded:
                self._ready = True
                logger.info("Loogle ready")
                return True
            # Check stderr for error messages
            try:
                stderr_data = await asyncio.wait_for(
                    self.process.stderr.read(), timeout=1
                )
                if stderr_data:
                    logger.error(f"Loogle stderr: {stderr_data.decode().strip()}")
            except asyncio.TimeoutError:
                pass
            logger.error(f"Loogle failed to start. stdout: {decoded.strip()}")
            return False
        except asyncio.TimeoutError:
            logger.error("Loogle startup timeout")
            return False
        except Exception as e:
            logger.error(f"Start failed: {e}")
            return False

    async def query(self, q: str, num_results: int = 8) -> list[dict[str, Any]]:
        async with self._lock:
            # Try up to 2 attempts (initial + one restart)
            for attempt in range(2):
                if (
                    not self._ready
                    or self.process is None
                    or self.process.returncode is not None
                ):
                    if attempt > 0:
                        raise RuntimeError("Loogle subprocess not ready")
                    self._ready = False
                    if not await self.start():
                        raise RuntimeError("Failed to start loogle")
                    continue

                try:
                    self.process.stdin.write(f"{q}\n".encode())
                    await self.process.stdin.drain()
                    line = await asyncio.wait_for(
                        self.process.stdout.readline(), timeout=30
                    )
                    response = json.loads(line.decode())
                    if err := response.get("error"):
                        logger.warning(f"Query error: {err}")
                        return []
                    return [
                        {
                            "name": h.get("name", ""),
                            "type": h.get("type", ""),
                            "module": h.get("module", ""),
                            "doc": h.get("doc"),
                        }
                        for h in response.get("hits", [])[:num_results]
                    ]
                except asyncio.TimeoutError:
                    raise RuntimeError("Query timeout") from None
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Invalid response: {e}") from e

            raise RuntimeError("Loogle subprocess not ready")

    async def stop(self) -> None:
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    pass
            except Exception:
                pass
            self.process = None
            self._ready = False

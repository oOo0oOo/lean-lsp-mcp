"""Loogle search - local subprocess and remote API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import ssl
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import certifi
import orjson

from lean_lsp_mcp.models import LoogleResult
from lean_lsp_mcp import config

# Re-exported for backward compatibility; canonical definition lives in config.
DEFAULT_LOOGLE_URL = config.DEFAULT_LOOGLE_URL

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    if d := config.loogle_cache_dir():
        return Path(d)
    if os.name == "nt":
        xdg = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(xdg) / "lean-lsp-mcp" / "loogle"


class LoogleQueryError(RuntimeError):
    """Loogle rejected the query itself (parse error, bad syntax)."""


def loogle_remote(query: str, num_results: int) -> list[LoogleResult] | str:
    """Query the remote loogle API.

    Set LOOGLE_URL to override the default endpoint.
    Set LOOGLE_HEADERS to a JSON object of extra headers (e.g. '{"X-API-Key": "..."}').
    """
    base = config.loogle_url()
    try:
        headers = {"User-Agent": "lean-lsp-mcp/0.1"}
        if extra := config.loogle_headers_raw():
            headers.update(json.loads(extra))
        req = urllib.request.Request(
            f"{base}/json?q={urllib.parse.quote(query)}",
            headers=headers,
        )
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as response:
            results = orjson.loads(response.read())
        if err := results.get("error"):
            suggestions = results.get("suggestions") or []
            hint = f" Suggestions: {', '.join(suggestions[:5])}" if suggestions else ""
            return f"loogle query error: {err}{hint}"
        hits = results.get("hits") or []
        hits = hits[:num_results]
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
    """Manages local loogle installation and async subprocess.

    Args:
        cache_dir: Directory for loogle repo and indices (default: ~/.cache/lean-lsp-mcp/loogle)
        project_path: Lean project whose Mathlib should be searched
    """

    REPO_URL = "https://github.com/nomeata/loogle.git"
    REPO_REF = "9f11169aaebf1ed1e7dcc4077f2aafe0fcf66fd0"
    READY_SIGNAL = "Loogle is ready."

    def __init__(self, cache_dir: Path | None = None, project_path: Path | None = None):
        self.cache_dir = cache_dir or get_cache_dir()
        self.index_dir = self.cache_dir / "index"
        self.project_path = (
            project_path.resolve(strict=False) if project_path is not None else None
        )
        self.process: asyncio.subprocess.Process | None = None
        self._ready = False
        self._lock = asyncio.Lock()

    def _get_project_toolchain(self) -> str | None:
        if self.project_path is None:
            return None
        try:
            return (
                (self.project_path / "lean-toolchain")
                .read_text(encoding="utf-8")
                .strip()
            )
        except OSError:
            return None

    @property
    def repo_dir(self) -> Path:
        toolchain = self._get_project_toolchain() or "unknown"
        key = hashlib.sha256(toolchain.encode()).hexdigest()[:12]
        return self.cache_dir / f"repo-{self.REPO_REF[:12]}-{key}"

    @property
    def build_dir(self) -> Path:
        return self.repo_dir / ".lake" / "build"

    @property
    def binary_path(self) -> Path:
        return self.build_dir / "bin" / "loogle"

    @property
    def index_path(self) -> Path:
        project = str(self.project_path or "unknown")
        key = hashlib.sha256(project.encode()).hexdigest()[:12]
        return self.index_dir / f"mathlib-{key}.idx"

    @property
    def is_installed(self) -> bool:
        return self._get_project_toolchain() is not None and self.binary_path.exists()

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
        self,
        cmd: list[str],
        timeout: int = 300,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        run_env = env if env is not None else os.environ.copy()
        run_env["LAKE_ARTIFACT_CACHE"] = "false"
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            cwd=cwd or self.repo_dir,
            env=run_env,
        )

    def _checkout_repo_ref(self) -> bool:
        try:
            current = self._run(
                ["git", "rev-parse", "HEAD"], cwd=self.repo_dir
            ).stdout.strip()
            if current != self.REPO_REF:
                fetched = self._run(
                    ["git", "fetch", "--depth", "1", "origin", self.REPO_REF],
                    cwd=self.repo_dir,
                )
                if fetched.returncode != 0:
                    logger.error(
                        "Loogle revision fetch failed: %s", fetched.stderr[:2000]
                    )
                    return False
            checked_out = self._run(
                ["git", "checkout", "--detach", self.REPO_REF],
                cwd=self.repo_dir,
            )
            if checked_out.returncode != 0:
                logger.error(
                    "Loogle revision checkout failed: %s", checked_out.stderr[:2000]
                )
                return False
            return True
        except Exception as exc:
            logger.error("Loogle revision setup failed: %s", exc)
            return False

    def _clone_repo(self) -> bool:
        if self.repo_dir.exists():
            return (self.repo_dir / ".git").exists() and self._checkout_repo_ref()
        logger.info(f"Cloning loogle to {self.repo_dir}...")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            r = self._run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--no-checkout",
                    self.REPO_URL,
                    str(self.repo_dir),
                ],
                cwd=self.cache_dir,
            )
            if r.returncode != 0:
                logger.error("Clone failed (exit %s).", r.returncode)
                if r.stdout:
                    logger.error("Clone stdout:\n%s", r.stdout[:2000])
                if r.stderr:
                    logger.error("Clone stderr:\n%s", r.stderr[:2000])
                return False
            return self._checkout_repo_ref()
        except OSError as e:
            logger.error("Clone setup error: %s", e)
            return False
        except Exception as e:
            logger.error(f"Clone error: {e}")
            return False

    def _build_loogle(self) -> bool:
        if self.is_installed:
            return True
        if not self.repo_dir.exists():
            return False
        toolchain = self._get_project_toolchain()
        if toolchain is None:
            return False
        logger.info("Building loogle for toolchain %s...", toolchain)
        try:
            result = self._run(
                ["lake", "build", "loogle"],
                timeout=900,
                env=self._project_env(),
            )
            if result.returncode != 0:
                logger.error("Build failed (exit %s).", result.returncode)
                if result.stdout:
                    logger.error("Build stdout:\n%s", result.stdout[:2000])
                if result.stderr:
                    logger.error("Build stderr:\n%s", result.stderr[:2000])
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Build error: {e}")
            return False

    def check_environment(self) -> tuple[bool, str]:
        """Check if the loogle environment is valid. Returns (ok, error_msg)."""
        if self.project_path is None:
            return False, "Lean project path not set"
        if self._get_project_toolchain() is None:
            return False, "Lean project has no readable lean-toolchain file"
        if not self.is_installed:
            return False, "Loogle binary not found"
        return True, ""

    def _project_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["LAKE_ARTIFACT_CACHE"] = "false"
        if toolchain := self._get_project_toolchain():
            env["ELAN_TOOLCHAIN"] = toolchain
        return env

    def set_project_path(self, project_path: Path | None) -> bool:
        """Update the active project. Returns whether the project changed."""
        resolved = (
            project_path.resolve(strict=False) if project_path is not None else None
        )
        changed = resolved != self.project_path
        self.project_path = resolved
        return changed

    def ensure_installed(self) -> bool:
        if self.project_path is None or self._get_project_toolchain() is None:
            logger.warning("A Lean project with a lean-toolchain file is required")
            return False
        ok, err = self._check_prerequisites()
        if not ok:
            logger.warning(f"Prerequisites: {err}")
            return False
        if not self._clone_repo():
            return False
        if not self._build_loogle():
            return False
        return self.is_installed

    async def start(self) -> bool:
        if self.process is not None and self.process.returncode is None:
            return self._ready
        if not self.is_installed and not await asyncio.to_thread(self.ensure_installed):
            return False
        ok, err = self.check_environment()
        if not ok:
            logger.error(f"Loogle environment check failed: {err}")
            return False

        self.index_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "lake",
            "env",
            str(self.binary_path),
            "--json",
            "--interactive",
            "--index-file",
            str(self.index_path),
        ]

        logger.info("Starting loogle for project %s...", self.project_path)
        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
                env=self._project_env(),
            )
            assert self.process.stdout is not None and self.process.stderr is not None
            # Loading a pre-built index is fast (~seconds), but Loogle's default
            # `use` mode builds a missing or stale Mathlib index first. Allow
            # for the initial build, which can take a couple of minutes.
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=300)
            decoded = line.decode("utf-8", errors="replace")
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
                    logger.error(
                        f"Loogle stderr: {stderr_data.decode('utf-8', errors='replace').strip()}"
                    )
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

    async def _desync_stop(self) -> None:
        """Kill the subprocess after a stream desync (timeout/garbage)."""
        self._ready = False
        if self.process is not None:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
            self.process = None

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

                assert self.process.stdin is not None
                assert self.process.stdout is not None
                # The protocol is one line per query/response; a newline in
                # the query would desync every subsequent read.
                sanitized = " ".join(q.splitlines())
                try:
                    self.process.stdin.write(f"{sanitized}\n".encode())
                    await self.process.stdin.drain()
                    line = await asyncio.wait_for(
                        self.process.stdout.readline(), timeout=30
                    )
                    response = json.loads(line.decode("utf-8", errors="replace"))
                    if err := response.get("error"):
                        suggestions = response.get("suggestions") or []
                        hint = (
                            f" Suggestions: {', '.join(suggestions[:5])}"
                            if suggestions
                            else ""
                        )
                        raise LoogleQueryError(f"loogle query error: {err}{hint}")
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
                    # The late response would be attributed to the NEXT query;
                    # kill the subprocess so a restart resyncs the stream.
                    await self._desync_stop()
                    raise RuntimeError("Query timeout") from None
                except json.JSONDecodeError as e:
                    await self._desync_stop()
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

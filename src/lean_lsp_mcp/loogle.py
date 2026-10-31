"""Loogle search - local subprocess and remote API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import signal
import ssl
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import certifi
import orjson
import psutil

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
    if config.loogle_local_only():
        return "Local Loogle is unavailable; remote fallback is disabled."
    base = config.loogle_url()
    try:
        headers = {"User-Agent": config.user_agent()}
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


async def _await_cleanup(task: asyncio.Task[None]) -> None:
    """Finish owned cleanup before propagating even repeated cancellation."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


async def _terminate_process(
    process: asyncio.subprocess.Process,
    *,
    force: bool = False,
) -> None:
    """Reap a launcher and its POSIX group, including surviving descendants."""
    group = process.pid if os.name == "posix" and isinstance(process.pid, int) else None
    if group is not None:
        try:
            owner = psutil.Process(process.pid)
            members = [owner, *owner.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            # Inspection is optional; denied or incomplete snapshots
            # cannot establish that every owned member has exited.
            members = []

        def exited(member: psutil.Process) -> bool:
            try:
                return (
                    not member.is_running() or member.status() == psutil.STATUS_ZOMBIE
                )
            except psutil.NoSuchProcess:
                return True
            except (psutil.AccessDenied, PermissionError):
                return False

        def signal_group(sig: int) -> bool:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                return False
            except PermissionError:
                # macOS can return EPERM instead of ESRCH once a
                # terminated group contains only dying/zombie tasks.
                # Suppress it only after checking the owned members.
                if members and all(exited(member) for member in members):
                    return False
                if sig == 0:
                    # During native exit teardown macOS may report
                    # EPERM while psutil still sees an idle member.
                    # Keep waiting within the graceful deadline.
                    return True
                raise
            return True

        if signal_group(signal.SIGKILL if force else signal.SIGTERM) and not force:
            deadline = asyncio.get_running_loop().time() + 5
            while signal_group(0):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    signal_group(signal.SIGKILL)
                    break
                await asyncio.sleep(min(0.05, remaining))
    elif process.returncode is None:
        try:
            process.kill() if force else process.terminate()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), timeout=2 if force else 5)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await asyncio.wait_for(process.wait(), timeout=2)


async def _spawn_process(*cmd: str, **kwargs: Any) -> asyncio.subprocess.Process:
    # Publish the spawn task before awaiting it: cancellation must not lose a
    # process whose creation has reached the OS but has not returned to Python.
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *cmd,
            start_new_session=os.name == "posix",
            **kwargs,
        )
    )
    try:
        return await asyncio.shield(spawn)
    except asyncio.CancelledError:

        async def cleanup() -> None:
            process = await spawn
            await _terminate_process(process, force=True)

        await _await_cleanup(asyncio.create_task(cleanup()))
        raise


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
        self._start_lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = b""
        self._stop_task: asyncio.Task[None] | None = None
        self._generation = 0
        self._install_task: asyncio.Task[bool] | None = None

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
        module = config.loogle_module()
        if module != "Mathlib":
            project += "\0" + module
        key = hashlib.sha256(project.encode()).hexdigest()[:12]
        return self.index_dir / f"{module.lower()}-{key}.idx"

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

    async def _run(
        self,
        cmd: list[str],
        timeout: int = 300,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        run_env = env if env is not None else os.environ.copy()
        run_env["LAKE_ARTIFACT_CACHE"] = "false"
        process = await _spawn_process(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or self.repo_dir,
            env=run_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
            assert process.returncode is not None
            return subprocess.CompletedProcess(
                cmd,
                process.returncode,
                stdout.decode("utf-8"),
                stderr.decode("utf-8"),
            )
        except asyncio.TimeoutError:
            raise subprocess.TimeoutExpired(cmd, timeout) from None
        finally:
            await _await_cleanup(
                asyncio.create_task(_terminate_process(process, force=True))
            )

    async def _checkout_repo_ref(self) -> bool:
        try:
            current = (
                await self._run(["git", "rev-parse", "HEAD"], cwd=self.repo_dir)
            ).stdout.strip()
            if current != self.REPO_REF:
                fetched = await self._run(
                    ["git", "fetch", "--depth", "1", "origin", self.REPO_REF],
                    cwd=self.repo_dir,
                )
                if fetched.returncode != 0:
                    logger.error(
                        "Loogle revision fetch failed: %s", fetched.stderr[:2000]
                    )
                    return False
            checked_out = await self._run(
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

    async def _clone_repo(self) -> bool:
        if self.repo_dir.exists():
            return (self.repo_dir / ".git").exists() and await self._checkout_repo_ref()
        logger.info(f"Cloning loogle to {self.repo_dir}...")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            r = await self._run(
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
            return await self._checkout_repo_ref()
        except OSError as e:
            logger.error("Clone setup error: %s", e)
            return False
        except Exception as e:
            logger.error(f"Clone error: {e}")
            return False

    async def _build_loogle(self) -> bool:
        if self.is_installed:
            return True
        if not self.repo_dir.exists():
            return False
        toolchain = self._get_project_toolchain()
        if toolchain is None:
            return False
        logger.info("Building loogle for toolchain %s...", toolchain)
        try:
            result = await self._run(
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
        """Install from a synchronous caller; server code uses async start()."""
        return asyncio.run(self._ensure_installed())

    async def _ensure_installed(self) -> bool:
        if self.project_path is None or self._get_project_toolchain() is None:
            logger.warning("A Lean project with a lean-toolchain file is required")
            return False
        # The cache key includes the pinned revision and project toolchain.
        # A completed installation needs neither a writable Git checkout nor
        # another fetch/build (and can be used while offline).
        if self.is_installed:
            return True
        ok, err = self._check_prerequisites()
        if not ok:
            logger.warning(f"Prerequisites: {err}")
            return False
        if not await self._clone_repo():
            return False
        if not await self._build_loogle():
            return False
        return self.is_installed

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        while chunk := await process.stderr.read(4096):
            self._stderr_tail = (self._stderr_tail + chunk)[-8192:]
            logger.debug(
                "Loogle stderr: %s", chunk.decode("utf-8", errors="replace").rstrip()
            )

    async def _wait_ready(self, process: asyncio.subprocess.Process) -> bool:
        assert process.stdout is not None
        while line := await process.stdout.readline():
            decoded = line.decode("utf-8", errors="replace")
            if self.READY_SIGNAL in decoded:
                return True
            logger.debug("Loogle startup: %s", decoded.rstrip())
        return False

    async def start(self) -> bool:
        # Startup is independent of the query lock because application setup
        # can also call start directly. Never overwrite a live child owner.
        async with self._start_lock:
            if self.is_running:
                return True
            if self.process is not None or self._stop_task is not None:
                await self.stop()
            generation = self._generation
            if not self.is_installed:
                install = asyncio.create_task(self._ensure_installed())
                self._install_task = install
                try:
                    installed = await asyncio.shield(install)
                except asyncio.CancelledError:
                    stopped = generation != self._generation
                    await self.stop()
                    if stopped:
                        return False
                    raise
                finally:
                    if self._install_task is install:
                        self._install_task = None
                if not installed:
                    return False
            if generation != self._generation:
                return False
            ok, err = self.check_environment()
            if not ok:
                logger.error("Loogle environment check failed: %s", err)
                return False

            self.index_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "lake",
                "env",
                str(self.binary_path),
                "--json",
                "--interactive",
                *(
                    ["--module", config.loogle_module()]
                    if config.loogle_module() != "Mathlib"
                    else []
                ),
                "--index-file",
                str(self.index_path),
            ]
            logger.info("Starting loogle for project %s...", self.project_path)
            self._stderr_tail = b""
            try:
                self.process = await _spawn_process(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.project_path,
                    env=self._project_env(),
                )
                process = self.process
                self._stderr_task = asyncio.create_task(self._drain_stderr(process))
                if generation != self._generation:
                    await self.stop()
                    return False
                # A fresh Mathlib index is substantially slower than reading
                # the reusable on-disk cache; both use the native use-index mode.
                ready = await asyncio.wait_for(self._wait_ready(process), timeout=300)
                self._ready = (
                    ready and self.process is process and generation == self._generation
                )
                if self._ready:
                    logger.info("Loogle ready")
                    return True
                logger.error("Loogle exited before ready")
            except asyncio.TimeoutError:
                logger.error("Loogle startup timeout")
            except asyncio.CancelledError:
                await self.stop()
                raise
            except Exception as exc:
                logger.error("Start failed: %s", exc)
            await self.stop()
            if self._stderr_tail:
                logger.error(
                    "Loogle stderr: %s",
                    self._stderr_tail.decode("utf-8", errors="replace").rstrip(),
                )
            return False

    async def _desync_stop(self) -> None:
        """Kill and reap the owned process after stream desynchronization."""
        await self._stop_owned(force=True)

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
                except asyncio.CancelledError:
                    # A cancelled read can leave the old response queued for
                    # the next query, just as a timeout does.
                    await self._desync_stop()
                    raise
                except asyncio.TimeoutError:
                    # The late response would be attributed to the NEXT query;
                    # kill the subprocess so a restart resyncs the stream.
                    await self._desync_stop()
                    raise RuntimeError("Query timeout") from None
                except json.JSONDecodeError as e:
                    await self._desync_stop()
                    raise RuntimeError(f"Invalid response: {e}") from e

            raise RuntimeError("Loogle subprocess not ready")

    async def _stop_process(self, *, force: bool = False) -> None:
        install = self._install_task
        if install is not None:
            install.cancel()
            await asyncio.gather(install, return_exceptions=True)
            if self._install_task is install:
                self._install_task = None
        process, self.process = self.process, None
        drain, self._stderr_task = self._stderr_task, None
        self._ready = False
        try:
            if process is not None:
                await _terminate_process(process, force=force)
        finally:
            if drain is not None:
                if not drain.done():
                    drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)

    async def _stop_owned(self, *, force: bool = False) -> None:
        self._generation += 1
        cancelled = False
        while True:
            cleanup = self._stop_task
            if cleanup is None:
                cleanup = asyncio.create_task(self._stop_process(force=force))
                self._stop_task = cleanup
            try:
                try:
                    await _await_cleanup(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            finally:
                if cleanup.done() and self._stop_task is cleanup:
                    self._stop_task = None
            # A spawn may finish while a previous empty cleanup task is still
            # retained. Rejoining that task must also drain the newly-published
            # process rather than mistakenly return with an unowned child.
            if (
                self.process is None
                and self._stderr_task is None
                and self._install_task is None
            ):
                break
        if cancelled:
            raise asyncio.CancelledError

    async def stop(self) -> None:
        """Stop and reap the Lake launcher and its owned native Loogle child."""
        await self._stop_owned()

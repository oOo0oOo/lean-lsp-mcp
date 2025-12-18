"""Lean REPL subprocess manager.

Manages per-project REPL subprocesses with session tracking for
environment and proof state isolation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from lean_lsp_mcp.repl_models import (
    ReplCommandRequest,
    ReplCommandResponse,
    ReplFileRequest,
    ReplMessage,
    ReplPickleEnvRequest,
    ReplPosition,
    ReplRequest,
    ReplSorry,
    ReplTacticInfo,
    ReplTacticRequest,
    ReplTacticResponse,
    ReplUnpickleEnvRequest,
)

logger = logging.getLogger(__name__)


def get_repl_cache_dir() -> Path:
    """Get the cache directory for REPL repo and binaries."""
    if d := os.environ.get("LEAN_REPL_CACHE_DIR"):
        return Path(d)
    xdg = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(xdg) / "lean-lsp-mcp" / "repl"


def get_repl_timeout() -> float:
    """Get the command timeout in seconds."""
    try:
        return float(os.environ.get("LEAN_REPL_TIMEOUT", "60"))
    except ValueError:
        return 60.0


@dataclass
class ReplSession:
    """Tracks state for a single REPL session within a project."""

    session_id: str
    project_path: Path
    current_env: int | None = None
    proof_states: dict[int, dict[str, Any]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def update_env(self, env: int | None) -> None:
        """Update the current environment ID."""
        if env is not None:
            self.current_env = env

    def add_proof_states(self, sorries: list[ReplSorry]) -> None:
        """Track proof states from sorries."""
        for sorry in sorries:
            self.proof_states[sorry.proofState] = {
                "goal": sorry.goal,
                "pos": sorry.pos.model_dump(),
            }


class ReplManager:
    """Manages per-project REPL subprocesses with session tracking.

    Features:
    - Per-project subprocess lifecycle management
    - Session-based environment isolation
    - JSON protocol communication
    - Auto-discovery of project-local or global REPL binary

    Usage:
        manager = ReplManager()
        await manager.start_for_project(project_path)
        response = await manager.send_command(project_path, ReplCommandRequest(cmd="def x := 1"))
    """

    REPO_URL = "https://github.com/leanprover-community/repl.git"
    DEFAULT_SESSION = "_default"

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or get_repl_cache_dir()
        self.repo_dir = self.cache_dir / "repo"

        # Per-project state
        self._processes: dict[Path, asyncio.subprocess.Process] = {}
        self._locks: dict[Path, asyncio.Lock] = {}

        # Session state (keyed by session_id)
        self._sessions: dict[str, ReplSession] = {}
        self._project_sessions: dict[Path, list[str]] = {}

        # Global lock for process management
        self._global_lock = asyncio.Lock()

    @property
    def global_binary_path(self) -> Path:
        """Path to the globally built REPL binary."""
        return self.repo_dir / ".lake" / "build" / "bin" / "repl"

    @property
    def is_global_installed(self) -> bool:
        """Check if global REPL is built."""
        return self.global_binary_path.exists()

    # =========================================================================
    # Binary Discovery and Building
    # =========================================================================

    def _get_project_repl_binary(self, project_path: Path) -> Path | None:
        """Check if project has REPL built locally."""
        local_repl = project_path / ".lake" / "build" / "bin" / "repl"
        if local_repl.exists():
            return local_repl
        return None

    def _check_prerequisites(self) -> tuple[bool, str]:
        """Check if git and lake are available."""
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
        """Run a subprocess synchronously."""
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
        """Clone the REPL repository."""
        if self.repo_dir.exists():
            return True
        logger.info(f"Cloning REPL to {self.repo_dir}...")
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

    def _build_global_repl(self) -> bool:
        """Build the global REPL binary."""
        if self.is_global_installed:
            return True
        if not self.repo_dir.exists():
            return False
        logger.info("Building REPL (this may take a few minutes)...")
        try:
            # Try to get mathlib cache
            self._run(["lake", "exe", "cache", "get"], timeout=600)
        except Exception as e:
            logger.warning(f"Cache download: {e}")
        try:
            return self._run(["lake", "build"], timeout=900).returncode == 0
        except Exception as e:
            logger.error(f"Build error: {e}")
            return False

    async def ensure_repl_for_project(self, project_path: Path) -> Path | None:
        """Ensure REPL is available for a project.

        Returns the path to the REPL binary, or None if unavailable.
        Tries project-local first, then falls back to global.
        """
        # Try project-local first
        if local := self._get_project_repl_binary(project_path):
            logger.debug(f"Using project-local REPL at {local}")
            return local

        # Fall back to global
        if not self.is_global_installed:
            ok, err = self._check_prerequisites()
            if not ok:
                logger.warning(f"Prerequisites: {err}")
                return None
            if not self._clone_repo():
                return None
            if not self._build_global_repl():
                return None

        return self.global_binary_path

    # =========================================================================
    # Process Lifecycle
    # =========================================================================

    async def start_for_project(self, project_path: Path) -> bool:
        """Start or reuse REPL subprocess for a project."""
        async with self._global_lock:
            # Check if already running
            if project_path in self._processes:
                proc = self._processes[project_path]
                if proc.returncode is None:
                    return True
                # Process died, clean up
                del self._processes[project_path]
                if project_path in self._locks:
                    del self._locks[project_path]

            repl_binary = await self.ensure_repl_for_project(project_path)
            if not repl_binary:
                return False

            # Build command - use lake env if global binary for correct LEAN_PATH
            if repl_binary == self.global_binary_path:
                cmd = ["lake", "env", str(repl_binary)]
            else:
                cmd = [str(repl_binary)]

            logger.info(f"Starting REPL for {project_path}...")
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=project_path,
                )
                self._processes[project_path] = proc
                self._locks[project_path] = asyncio.Lock()
                logger.info(f"REPL started for {project_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to start REPL: {e}")
                return False

    async def stop_for_project(self, project_path: Path) -> None:
        """Stop REPL subprocess for a project."""
        async with self._global_lock:
            if project_path not in self._processes:
                return

            proc = self._processes[project_path]
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2)
                except asyncio.TimeoutError:
                    pass
            except Exception:
                pass

            del self._processes[project_path]
            if project_path in self._locks:
                del self._locks[project_path]

            # Clean up sessions for this project
            if project_path in self._project_sessions:
                for session_id in self._project_sessions[project_path]:
                    if session_id in self._sessions:
                        del self._sessions[session_id]
                del self._project_sessions[project_path]

    async def stop_all(self) -> None:
        """Stop all REPL subprocesses."""
        projects = list(self._processes.keys())
        for project_path in projects:
            await self.stop_for_project(project_path)

    # =========================================================================
    # Session Management
    # =========================================================================

    def create_session(self, project_path: Path) -> ReplSession:
        """Create a new session for a project."""
        session_id = str(uuid.uuid4())
        session = ReplSession(session_id=session_id, project_path=project_path)
        self._sessions[session_id] = session

        if project_path not in self._project_sessions:
            self._project_sessions[project_path] = []
        self._project_sessions[project_path].append(session_id)

        return session

    def get_session(self, session_id: str) -> ReplSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_or_create_default_session(self, project_path: Path) -> ReplSession:
        """Get or create the default session for a project."""
        default_id = f"{self.DEFAULT_SESSION}:{project_path}"
        if default_id not in self._sessions:
            session = ReplSession(session_id=default_id, project_path=project_path)
            self._sessions[default_id] = session
            if project_path not in self._project_sessions:
                self._project_sessions[project_path] = []
            self._project_sessions[project_path].append(default_id)
        return self._sessions[default_id]

    def list_sessions(self, project_path: Path | None = None) -> list[ReplSession]:
        """List sessions, optionally filtered by project."""
        if project_path is None:
            return list(self._sessions.values())
        return [
            self._sessions[sid]
            for sid in self._project_sessions.get(project_path, [])
            if sid in self._sessions
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]
        del self._sessions[session_id]

        project_path = session.project_path
        if project_path in self._project_sessions:
            try:
                self._project_sessions[project_path].remove(session_id)
            except ValueError:
                pass

        return True

    # =========================================================================
    # Command Execution
    # =========================================================================

    def _serialize_request(self, request: ReplRequest) -> str:
        """Serialize a request to JSON."""
        # Use model_dump with exclude_none to skip None fields
        data = request.model_dump(exclude_none=True)
        return orjson.dumps(data).decode() + "\n"

    def _parse_command_response(self, data: dict[str, Any]) -> ReplCommandResponse:
        """Parse a command response from JSON."""
        # Handle error responses
        if "error" in data:
            return ReplCommandResponse(
                env=None,
                sorries=[],
                messages=[
                    ReplMessage(
                        severity="error",
                        pos=ReplPosition(line=0, column=0),
                        data=data["error"],
                    )
                ],
                tactics=[],
            )

        # Parse sorries
        sorries = []
        for s in data.get("sorries", []):
            sorries.append(
                ReplSorry(
                    pos=ReplPosition(**s["pos"]),
                    goal=s.get("goal", ""),
                    proofState=s.get("proofState", 0),
                    endPos=ReplPosition(**s["endPos"]) if "endPos" in s else None,
                )
            )

        # Parse messages
        messages = []
        for m in data.get("messages", []):
            messages.append(
                ReplMessage(
                    severity=m.get("severity", "error"),
                    pos=ReplPosition(**m["pos"]) if "pos" in m else ReplPosition(line=0, column=0),
                    endPos=ReplPosition(**m["endPos"]) if "endPos" in m else None,
                    data=m.get("data", ""),
                )
            )

        # Parse tactics
        tactics = []
        for t in data.get("tactics", []):
            tactics.append(
                ReplTacticInfo(
                    tactic=t.get("tactic", ""),
                    goals=t.get("goals", []),
                    pos=ReplPosition(**t["pos"]) if "pos" in t else ReplPosition(line=0, column=0),
                    endPos=ReplPosition(**t["endPos"]) if "endPos" in t else None,
                )
            )

        return ReplCommandResponse(
            env=data.get("env"),
            sorries=sorries,
            messages=messages,
            tactics=tactics,
        )

    def _parse_tactic_response(self, data: dict[str, Any]) -> ReplTacticResponse:
        """Parse a tactic response from JSON."""
        # Handle error responses
        if "error" in data:
            return ReplTacticResponse(
                proofState=None,
                goals=[],
                messages=[
                    ReplMessage(
                        severity="error",
                        pos=ReplPosition(line=0, column=0),
                        data=data["error"],
                    )
                ],
                traces=[],
                proofStatus=None,
            )

        # Parse messages if present
        messages = []
        for m in data.get("messages", []):
            messages.append(
                ReplMessage(
                    severity=m.get("severity", "info"),
                    pos=ReplPosition(**m["pos"]) if "pos" in m else ReplPosition(line=0, column=0),
                    endPos=ReplPosition(**m["endPos"]) if "endPos" in m else None,
                    data=m.get("data", ""),
                )
            )

        return ReplTacticResponse(
            proofState=data.get("proofState"),
            goals=data.get("goals", []),
            messages=messages,
            traces=data.get("traces", []),
            proofStatus=data.get("proofStatus"),
        )

    async def send_command(
        self,
        project_path: Path,
        request: ReplRequest,
        session_id: str | None = None,
    ) -> ReplCommandResponse | ReplTacticResponse:
        """Send a command to the REPL and parse response.

        Args:
            project_path: Path to the Lean project
            request: The REPL request to send
            session_id: Optional session ID for state tracking

        Returns:
            Parsed response from the REPL
        """
        # Get or create session
        if session_id:
            session = self.get_session(session_id)
            if session is None:
                session = self.create_session(project_path)
                # Preserve the requested session_id
                self._sessions[session_id] = session
                self._sessions[session_id].session_id = session_id
                del self._sessions[session.session_id]
                session = self._sessions[session_id]
        else:
            session = self.get_or_create_default_session(project_path)

        # Ensure REPL is running
        if not await self.start_for_project(project_path):
            return ReplCommandResponse(
                env=None,
                sorries=[],
                messages=[
                    ReplMessage(
                        severity="error",
                        pos=ReplPosition(line=0, column=0),
                        data="Failed to start REPL subprocess",
                    )
                ],
                tactics=[],
            )

        proc = self._processes[project_path]
        lock = self._locks[project_path]
        timeout = get_repl_timeout()

        async with lock:
            if proc.returncode is not None:
                return ReplCommandResponse(
                    env=None,
                    sorries=[],
                    messages=[
                        ReplMessage(
                            severity="error",
                            pos=ReplPosition(line=0, column=0),
                            data="REPL process has terminated",
                        )
                    ],
                    tactics=[],
                )

            try:
                # Send request
                request_json = self._serialize_request(request)
                proc.stdin.write(request_json.encode())
                await proc.stdin.drain()

                # Read response (single JSON line followed by blank line)
                response_line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=timeout
                )

                # Skip blank line if present
                if response_line.strip() == b"":
                    response_line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=timeout
                    )

                response_data = orjson.loads(response_line)

                # Check if this is a tactic response
                if isinstance(request, ReplTacticRequest):
                    response = self._parse_tactic_response(response_data)
                else:
                    response = self._parse_command_response(response_data)

                # Update session state
                if isinstance(response, ReplCommandResponse):
                    session.update_env(response.env)
                    session.add_proof_states(response.sorries)

                return response

            except asyncio.TimeoutError:
                return ReplCommandResponse(
                    env=None,
                    sorries=[],
                    messages=[
                        ReplMessage(
                            severity="error",
                            pos=ReplPosition(line=0, column=0),
                            data=f"REPL command timeout after {timeout}s",
                        )
                    ],
                    tactics=[],
                )
            except orjson.JSONDecodeError as e:
                return ReplCommandResponse(
                    env=None,
                    sorries=[],
                    messages=[
                        ReplMessage(
                            severity="error",
                            pos=ReplPosition(line=0, column=0),
                            data=f"Invalid REPL response: {e}",
                        )
                    ],
                    tactics=[],
                )
            except Exception as e:
                return ReplCommandResponse(
                    env=None,
                    sorries=[],
                    messages=[
                        ReplMessage(
                            severity="error",
                            pos=ReplPosition(line=0, column=0),
                            data=f"REPL error: {e}",
                        )
                    ],
                    tactics=[],
                )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def run_command(
        self,
        project_path: Path,
        cmd: str,
        env: int | None = None,
        session_id: str | None = None,
    ) -> ReplCommandResponse:
        """Run a Lean command in the REPL."""
        request = ReplCommandRequest(cmd=cmd, env=env)
        response = await self.send_command(project_path, request, session_id)
        if isinstance(response, ReplCommandResponse):
            return response
        # Should not happen, but handle gracefully
        return ReplCommandResponse(
            env=None,
            sorries=[],
            messages=response.messages,
            tactics=[],
        )

    async def run_tactic(
        self,
        project_path: Path,
        tactic: str,
        proof_state: int,
        session_id: str | None = None,
    ) -> ReplTacticResponse:
        """Apply a tactic in the REPL."""
        request = ReplTacticRequest(tactic=tactic, proofState=proof_state)
        response = await self.send_command(project_path, request, session_id)
        if isinstance(response, ReplTacticResponse):
            return response
        # Convert command response to tactic response
        return ReplTacticResponse(
            proofState=None,
            goals=[],
            messages=response.messages if isinstance(response, ReplCommandResponse) else [],
            traces=[],
            proofStatus=None,
        )

    async def load_file(
        self,
        project_path: Path,
        file_path: str,
        all_tactics: bool = False,
        session_id: str | None = None,
    ) -> ReplCommandResponse:
        """Load a Lean file into the REPL."""
        request = ReplFileRequest(path=file_path, allTactics=all_tactics)
        response = await self.send_command(project_path, request, session_id)
        if isinstance(response, ReplCommandResponse):
            return response
        return ReplCommandResponse(
            env=None,
            sorries=[],
            messages=response.messages,
            tactics=[],
        )

    async def pickle_env(
        self,
        project_path: Path,
        env: int,
        path: str,
        session_id: str | None = None,
    ) -> ReplCommandResponse:
        """Pickle an environment to a file."""
        request = ReplPickleEnvRequest(pickleTo=path, env=env)
        response = await self.send_command(project_path, request, session_id)
        if isinstance(response, ReplCommandResponse):
            return response
        return ReplCommandResponse(
            env=None,
            sorries=[],
            messages=response.messages,
            tactics=[],
        )

    async def unpickle_env(
        self,
        project_path: Path,
        path: str,
        session_id: str | None = None,
    ) -> ReplCommandResponse:
        """Unpickle an environment from a file."""
        request = ReplUnpickleEnvRequest(unpickleEnvFrom=path)
        response = await self.send_command(project_path, request, session_id)
        if isinstance(response, ReplCommandResponse):
            return response
        return ReplCommandResponse(
            env=None,
            sorries=[],
            messages=response.messages,
            tactics=[],
        )

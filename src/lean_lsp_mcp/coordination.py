from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import signal
import socket
import socketserver
import stat
import subprocess
import sys
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COORDINATION_MODE_DIRECT = "direct"
COORDINATION_MODE_BROKER = "broker"

DEFAULT_MAX_LINEAGE_DEPTH = 3
DEFAULT_MAX_WORKERS = 2
DEFAULT_LINEAGE_WORKER_LIMIT = 1
DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 10.0
# AF_UNIX path limits vary by platform (~104 bytes on macOS, 108 on Linux).
# Keep a little headroom to avoid edge-case bind/connect failures.
MAX_UNIX_SOCKET_PATH_BYTES = 96

ENV_COORDINATION_MODE = "LEAN_LSP_MCP_COORDINATION"
ENV_COORDINATION_DIR = "LEAN_LSP_MCP_COORDINATION_DIR"
ENV_MAX_LINEAGE_DEPTH = "LEAN_LSP_MCP_MAX_LINEAGE_DEPTH"
ENV_MAX_WORKERS = "LEAN_LSP_MCP_MAX_WORKERS"
ENV_INSTANCE_ID = "LEAN_LSP_MCP_INSTANCE_ID"
ENV_LINEAGE_ROOT = "LEAN_LSP_MCP_LINEAGE_ROOT"
ENV_LINEAGE_DEPTH = "LEAN_LSP_MCP_LINEAGE_DEPTH"


class CoordinationError(RuntimeError):
    """Raised when coordination broker operations fail."""


def default_coordination_dir() -> Path:
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    return tmp / "lean-lsp-mcp-coordination"


def broker_socket_path(coordination_dir: Path) -> Path:
    candidate = coordination_dir / "broker.sock"
    if os.name == "nt":
        return candidate
    if len(os.fsencode(str(candidate))) <= MAX_UNIX_SOCKET_PATH_BYTES:
        return candidate

    digest = hashlib.sha256(str(coordination_dir).encode("utf-8")).hexdigest()[:16]
    compact_dir = default_coordination_dir() / "sockets"
    compact_path = compact_dir / f"broker-{digest}.sock"
    if len(os.fsencode(str(compact_path))) <= MAX_UNIX_SOCKET_PATH_BYTES:
        return compact_path

    # Last-resort fallback for unusually long TMPDIR paths.
    return Path("/tmp") / f"lean-lsp-mcp-{digest}.sock"


def parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise CoordinationError(f"{name} must be an integer, got: {raw!r}") from exc
    return parsed


def parse_non_negative_int_env(name: str, default: int) -> int:
    value = parse_int_env(name, default)
    if value < 0:
        raise CoordinationError(f"{name} must be >= 0, got: {value}")
    return value


def derive_lineage(instance_id: str) -> tuple[str, int]:
    inherited_root = os.environ.get(ENV_LINEAGE_ROOT, "").strip()
    inherited_depth_raw = os.environ.get(ENV_LINEAGE_DEPTH, "").strip()
    if not inherited_root:
        return instance_id, 0
    try:
        parent_depth = int(inherited_depth_raw) if inherited_depth_raw else 0
    except ValueError as exc:
        raise CoordinationError(
            f"{ENV_LINEAGE_DEPTH} must be an integer, got: {inherited_depth_raw!r}"
        ) from exc
    if parent_depth < 0:
        raise CoordinationError(
            f"{ENV_LINEAGE_DEPTH} must be >= 0, got: {parent_depth}"
        )
    return inherited_root, parent_depth + 1


@dataclass(slots=True)
class InstanceRegistration:
    instance_id: str
    lineage_root: str
    lineage_depth: int
    max_lineage_depth: int
    pid: int
    updated_at: float


@dataclass(slots=True)
class LeaseRecord:
    lease_id: str
    instance_id: str
    lineage_root: str
    worker_key: str
    created_at: float


class _BrokerState:
    def __init__(self, max_workers: int, lineage_worker_limit: int) -> None:
        self.max_workers = max_workers
        # Never let lineage throttling undercut the configured global capacity.
        self.lineage_worker_limit = max(lineage_worker_limit, max_workers)
        self.instances: dict[str, InstanceRegistration] = {}
        self.leases: dict[str, LeaseRecord] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _cleanup_dead_instances_locked(self) -> None:
        dead = [
            instance_id
            for instance_id, registration in self.instances.items()
            if not self._pid_alive(registration.pid)
        ]
        if not dead:
            return

        dead_set = set(dead)
        for instance_id in dead:
            self.instances.pop(instance_id, None)

        for lease_id in [
            lease_id
            for lease_id, lease in self.leases.items()
            if lease.instance_id in dead_set
        ]:
            self.leases.pop(lease_id, None)

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        instance_id = str(payload["instance_id"])
        lineage_root = str(payload["lineage_root"])
        lineage_depth = int(payload["lineage_depth"])
        max_lineage_depth = int(payload["max_lineage_depth"])
        pid = int(payload["pid"])

        if lineage_depth < 0:
            raise CoordinationError(
                f"Lineage depth {lineage_depth} must be >= 0."
            )
        if lineage_depth > max_lineage_depth:
            raise CoordinationError(
                f"Lineage depth {lineage_depth} exceeds configured limit {max_lineage_depth}."
            )

        now = time.time()
        with self._lock:
            self._cleanup_dead_instances_locked()
            self.instances[instance_id] = InstanceRegistration(
                instance_id=instance_id,
                lineage_root=lineage_root,
                lineage_depth=lineage_depth,
                max_lineage_depth=max_lineage_depth,
                pid=pid,
                updated_at=now,
            )
        return {"registered": True}

    def unregister(self, payload: dict[str, Any]) -> dict[str, Any]:
        instance_id = str(payload["instance_id"])
        with self._lock:
            self.instances.pop(instance_id, None)
            for lease_id in [
                lease_id
                for lease_id, lease in self.leases.items()
                if lease.instance_id == instance_id
            ]:
                self.leases.pop(lease_id, None)
        return {"unregistered": True}

    def acquire(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        instance_id = str(payload["instance_id"])
        lineage_root = str(payload["lineage_root"])
        worker_key = str(payload["worker_key"])

        now = time.time()
        with self._lock:
            self._cleanup_dead_instances_locked()

            registration = self.instances.get(instance_id)
            if registration is None:
                raise CoordinationError("Instance is not registered with broker.")
            registration.updated_at = now

            # Idempotent acquire for same instance+worker.
            for lease in self.leases.values():
                if lease.instance_id == instance_id and lease.worker_key == worker_key:
                    return {"lease_id": lease.lease_id, "reused": True}, False

            # A single MCP instance should only hold one worker lease.
            if any(lease.instance_id == instance_id for lease in self.leases.values()):
                return {
                    "error": "Instance already holds a lease for another worker.",
                    "retryable": False,
                }, True

            if any(lease.worker_key == worker_key for lease in self.leases.values()):
                return {
                    "error": "Requested worker is busy.",
                    "retryable": True,
                }, True

            if len(self.leases) >= self.max_workers:
                return {
                    "error": f"Global worker limit reached ({self.max_workers}).",
                    "retryable": True,
                }, True

            lineage_count = sum(
                1 for lease in self.leases.values() if lease.lineage_root == lineage_root
            )
            if lineage_count >= self.lineage_worker_limit:
                return {
                    "error": (
                        "Lineage worker limit reached "
                        f"({self.lineage_worker_limit}) for lineage {lineage_root}."
                    ),
                    "retryable": True,
                }, True

            lease_id = uuid.uuid4().hex
            self.leases[lease_id] = LeaseRecord(
                lease_id=lease_id,
                instance_id=instance_id,
                lineage_root=lineage_root,
                worker_key=worker_key,
                created_at=now,
            )

            return {"lease_id": lease_id, "reused": False}, False

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        instance_id = str(payload["instance_id"])
        lease_id = str(payload["lease_id"])
        with self._lock:
            lease = self.leases.get(lease_id)
            if lease is None:
                return {"released": False, "missing": True}
            if lease.instance_id != instance_id:
                raise CoordinationError("Lease ownership mismatch.")
            self.leases.pop(lease_id, None)
        return {"released": True}

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._cleanup_dead_instances_locked()
            return {
                "instances": len(self.instances),
                "leases": len(self.leases),
                "max_workers": self.max_workers,
                "lineage_worker_limit": self.lineage_worker_limit,
            }


class _BrokerRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline()
        if not raw:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._write({"ok": False, "error": "Invalid JSON payload.", "retryable": False})
            return

        action = payload.get("action")
        state: _BrokerState = self.server.state  # type: ignore[attr-defined]
        try:
            if action == "ping":
                result = {"pong": True}
            elif action == "status":
                result = state.status()
            elif action == "register":
                result = state.register(payload)
            elif action == "unregister":
                result = state.unregister(payload)
            elif action == "acquire":
                result, is_error = state.acquire(payload)
                if is_error:
                    self._write(
                        {
                            "ok": False,
                            "error": result["error"],
                            "retryable": bool(result.get("retryable", False)),
                        }
                    )
                    return
            elif action == "release":
                result = state.release(payload)
            elif action == "shutdown":
                result = {"shutting_down": True}
                self._write({"ok": True, "result": result})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            else:
                self._write(
                    {
                        "ok": False,
                        "error": f"Unknown action: {action!r}",
                        "retryable": False,
                    }
                )
                return
        except CoordinationError as exc:
            self._write({"ok": False, "error": str(exc), "retryable": False})
            return
        except Exception:
            self._write({"ok": False, "error": "Broker internal error.", "retryable": False})
            return

        self._write({"ok": True, "result": result})

    def _write(self, payload: dict[str, Any]) -> None:
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        self.wfile.write(encoded)
        self.wfile.flush()


class _ThreadingUnixStreamServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def _probe_broker_socket_liveness(
    socket_path: Path, timeout_seconds: float = 0.2
) -> bool | None:
    """Probe whether a broker is definitely live on a Unix socket path.

    Returns:
        True: a broker ping succeeded.
        False: no live broker is reachable at the path.
        None: broker liveness is uncertain (do not unlink path).
    """
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_seconds)
            sock.connect(str(socket_path))
            sock.sendall(b'{"action":"ping"}\n')
            response = b""
            while not response.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
    except FileNotFoundError:
        return False
    except ConnectionRefusedError:
        return False
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ECONNREFUSED, errno.ENOTSOCK}:
            return False
        # Timeout or other IO issues should be treated as uncertain to avoid
        # unlinking a potentially live broker socket.
        return None

    if not response:
        return None

    try:
        payload = json.loads(response.decode("utf-8"))
    except json.JSONDecodeError:
        return None

    result = payload.get("result")
    if payload.get("ok") is True and isinstance(result, dict) and result.get("pong") is True:
        return True
    return None


def _run_broker_process(
    socket_path: Path, max_workers: int, lineage_worker_limit: int
) -> int:
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    state = _BrokerState(
        max_workers=max_workers, lineage_worker_limit=lineage_worker_limit
    )
    try:
        server = _ThreadingUnixStreamServer(str(socket_path), _BrokerRequestHandler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE or not socket_path.exists():
            raise

        live = _probe_broker_socket_liveness(socket_path)
        if live is False:
            try:
                mode = socket_path.stat().st_mode
            except FileNotFoundError:
                # The path disappeared between probe and retry.
                server = _ThreadingUnixStreamServer(str(socket_path), _BrokerRequestHandler)
            else:
                # Only remove confirmed stale socket inodes.
                if not stat.S_ISSOCK(mode):
                    raise
                with suppress(OSError):
                    socket_path.unlink()
                server = _ThreadingUnixStreamServer(str(socket_path), _BrokerRequestHandler)
        else:
            # Existing socket appears live or uncertain: fail fast without unlinking.
            raise
    server.state = state  # type: ignore[attr-defined]

    # Restrict local socket access to current user.
    with suppress(OSError):
        os.chmod(socket_path, 0o600)

    shutdown_event = threading.Event()

    def _shutdown_handler(_signum: int, _frame: object) -> None:
        if shutdown_event.is_set():
            return
        shutdown_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    previous_sigint = signal.signal(signal.SIGINT, _shutdown_handler)
    previous_sigterm = signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.server_close()
        with suppress(OSError):
            socket_path.unlink()

    return 0


class CoordinationClient:
    def __init__(
        self,
        *,
        coordination_dir: Path,
        max_workers: int,
        lineage_worker_limit: int = DEFAULT_LINEAGE_WORKER_LIMIT,
    ) -> None:
        self.coordination_dir = coordination_dir
        self.socket_path = broker_socket_path(coordination_dir)
        self.max_workers = max_workers
        # Match broker-side normalization so live limit validation remains consistent.
        self.lineage_worker_limit = max(lineage_worker_limit, max_workers)
        self._io_lock = threading.Lock()

    def _request(self, payload: dict[str, Any], timeout_seconds: float = 2.0) -> dict[str, Any]:
        with self._io_lock:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(timeout_seconds)
                    sock.connect(str(self.socket_path))
                    encoded = (
                        json.dumps(payload, separators=(",", ":")) + "\n"
                    ).encode("utf-8")
                    sock.sendall(encoded)
                    response = b""
                    while not response.endswith(b"\n"):
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        response += chunk
            except FileNotFoundError as exc:
                raise CoordinationError(
                    f"Coordination broker socket not found: {self.socket_path}"
                ) from exc
            except OSError as exc:
                raise CoordinationError(f"Coordination broker connection failed: {exc}") from exc

        if not response:
            raise CoordinationError("Coordination broker returned empty response.")

        try:
            payload_out = json.loads(response.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise CoordinationError("Coordination broker returned invalid JSON.") from exc

        if not payload_out.get("ok", False):
            error = str(payload_out.get("error", "unknown broker error"))
            retryable = bool(payload_out.get("retryable", False))
            if retryable:
                raise CoordinationError(f"RETRYABLE:{error}")
            raise CoordinationError(error)

        result = payload_out.get("result")
        if not isinstance(result, dict):
            raise CoordinationError("Coordination broker returned malformed result.")
        return result

    def _spawn_broker(self) -> None:
        self.coordination_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "lean_lsp_mcp.coordination",
            "broker",
            "--socket-path",
            str(self.socket_path),
            "--max-workers",
            str(self.max_workers),
            "--lineage-worker-limit",
            str(self.lineage_worker_limit),
        ]
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _should_respawn_after_ping_error(self, exc: CoordinationError) -> bool:
        cause = exc.__cause__
        if isinstance(cause, FileNotFoundError):
            return True
        if isinstance(cause, OSError):
            return cause.errno in {
                errno.ENOENT,
                errno.ECONNREFUSED,
                errno.ENOTSOCK,
            }

        # If no OS-level error is available, only respawn on explicit missing-socket errors.
        return "socket not found" in str(exc).lower()

    def _validate_live_broker_limits(self) -> None:
        status = self._request({"action": "status"})
        broker_max_workers = status.get("max_workers")
        broker_lineage_limit = status.get("lineage_worker_limit")
        if not isinstance(broker_max_workers, int) or not isinstance(
            broker_lineage_limit, int
        ):
            raise CoordinationError(
                "Coordination broker returned malformed status for limit validation."
            )
        mismatch_parts: list[str] = []
        if broker_max_workers != self.max_workers:
            mismatch_parts.append(
                f"max_workers={broker_max_workers} (expected {self.max_workers})"
            )
        if broker_lineage_limit != self.lineage_worker_limit:
            mismatch_parts.append(
                "lineage_worker_limit="
                f"{broker_lineage_limit} (expected {self.lineage_worker_limit})"
            )
        if mismatch_parts:
            mismatch_summary = ", ".join(mismatch_parts)
            raise CoordinationError(
                "Live coordination broker configuration mismatch: "
                f"{mismatch_summary}. Restart the broker or use a different "
                "coordination directory."
            )

    def ensure_available(self, timeout_seconds: float = 5.0) -> None:
        try:
            self._request({"action": "ping"})
            self._validate_live_broker_limits()
            return
        except CoordinationError as exc:
            if not self._should_respawn_after_ping_error(exc):
                raise

        self._spawn_broker()
        deadline = time.time() + timeout_seconds
        last_error: CoordinationError | None = None
        while time.time() < deadline:
            try:
                self._request({"action": "ping"})
                self._validate_live_broker_limits()
                return
            except CoordinationError as exc:
                if not self._should_respawn_after_ping_error(exc):
                    raise
                last_error = exc
                time.sleep(0.05)
        if last_error is not None:
            raise CoordinationError(
                f"Coordination broker unavailable after startup: {last_error}"
            ) from last_error
        raise CoordinationError("Coordination broker unavailable after startup.")

    def register_instance(
        self,
        *,
        instance_id: str,
        lineage_root: str,
        lineage_depth: int,
        max_lineage_depth: int,
        pid: int,
    ) -> None:
        self._request(
            {
                "action": "register",
                "instance_id": instance_id,
                "lineage_root": lineage_root,
                "lineage_depth": lineage_depth,
                "max_lineage_depth": max_lineage_depth,
                "pid": pid,
            }
        )

    def unregister_instance(self, *, instance_id: str) -> None:
        self._request({"action": "unregister", "instance_id": instance_id})

    def acquire_lease(
        self,
        *,
        instance_id: str,
        lineage_root: str,
        worker_key: str,
        timeout_seconds: float = DEFAULT_ACQUIRE_TIMEOUT_SECONDS,
    ) -> str:
        deadline = time.time() + timeout_seconds
        retry_error: CoordinationError | None = None
        while True:
            try:
                result = self._request(
                    {
                        "action": "acquire",
                        "instance_id": instance_id,
                        "lineage_root": lineage_root,
                        "worker_key": worker_key,
                    }
                )
                lease_id = result.get("lease_id")
                if not isinstance(lease_id, str) or not lease_id:
                    raise CoordinationError("Broker returned invalid lease id.")
                return lease_id
            except CoordinationError as exc:
                msg = str(exc)
                if msg.startswith("RETRYABLE:") and time.time() < deadline:
                    retry_error = exc
                    time.sleep(0.05)
                    continue
                if msg.startswith("RETRYABLE:"):
                    detail = msg[len("RETRYABLE:") :]
                    raise CoordinationError(
                        f"Timed out waiting for coordination lease: {detail}"
                    ) from retry_error
                raise

    def release_lease(self, *, instance_id: str, lease_id: str) -> None:
        self._request(
            {
                "action": "release",
                "instance_id": instance_id,
                "lease_id": lease_id,
            }
        )

    def status(self) -> dict[str, Any]:
        return self._request({"action": "status"})

    def shutdown(self) -> None:
        self._request({"action": "shutdown"})


def _broker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Lean LSP MCP coordination broker")
    parser.add_argument("--socket-path", required=True, type=Path)
    parser.add_argument("--max-workers", required=True, type=int)
    parser.add_argument("--lineage-worker-limit", required=True, type=int)
    args = parser.parse_args(argv)

    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be > 0")
    if args.lineage_worker_limit <= 0:
        raise SystemExit("--lineage-worker-limit must be > 0")

    return _run_broker_process(
        socket_path=args.socket_path,
        max_workers=args.max_workers,
        lineage_worker_limit=args.lineage_worker_limit,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("Usage: python -m lean_lsp_mcp.coordination broker ...")
    command = args[0]
    if command != "broker":
        raise SystemExit(f"Unknown command: {command!r}")
    return _broker_main(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())

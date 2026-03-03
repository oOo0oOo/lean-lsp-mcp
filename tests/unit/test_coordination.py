from __future__ import annotations

import errno
import os
import socket
import time
from pathlib import Path

import pytest

from lean_lsp_mcp.coordination import (
    CoordinationClient,
    CoordinationError,
    ENV_LINEAGE_DEPTH,
    ENV_LINEAGE_ROOT,
    MAX_UNIX_SOCKET_PATH_BYTES,
    _run_broker_process,
    broker_socket_path,
    derive_lineage,
)


def test_derive_lineage_for_root_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_LINEAGE_ROOT, raising=False)
    monkeypatch.delenv(ENV_LINEAGE_DEPTH, raising=False)

    root, depth = derive_lineage("instance-1")
    assert root == "instance-1"
    assert depth == 0


def test_derive_lineage_for_child_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_LINEAGE_ROOT, "root-abc")
    monkeypatch.setenv(ENV_LINEAGE_DEPTH, "2")

    root, depth = derive_lineage("instance-2")
    assert root == "root-abc"
    assert depth == 3


def test_derive_lineage_rejects_negative_inherited_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LINEAGE_ROOT, "root-abc")
    monkeypatch.setenv(ENV_LINEAGE_DEPTH, "-1")

    with pytest.raises(CoordinationError, match="must be >= 0"):
        derive_lineage("instance-3")


@pytest.mark.skipif(os.name == "nt", reason="unix socket limits are unix only")
def test_broker_socket_path_is_short_for_long_coordination_dir(tmp_path: Path) -> None:
    very_long = tmp_path / ("nested-" * 20) / ("coord-" * 20)
    socket_path = broker_socket_path(very_long)
    assert len(os.fsencode(str(socket_path))) <= MAX_UNIX_SOCKET_PATH_BYTES


@pytest.mark.skipif(os.name == "nt", reason="broker mode is Unix only")
def test_coordination_broker_register_acquire_release(tmp_path: Path) -> None:
    coordination_dir = tmp_path / "coord"
    client = CoordinationClient(coordination_dir=coordination_dir, max_workers=1)

    client.ensure_available(timeout_seconds=5.0)
    try:
        client.register_instance(
            instance_id="inst-a",
            lineage_root="lineage-a",
            lineage_depth=0,
            max_lineage_depth=3,
            pid=os.getpid(),
        )
        lease = client.acquire_lease(
            instance_id="inst-a",
            lineage_root="lineage-a",
            worker_key="/tmp/project-a::tc=abc::repl=0",
            timeout_seconds=0.2,
        )
        assert lease

        client.register_instance(
            instance_id="inst-b",
            lineage_root="lineage-a",
            lineage_depth=1,
            max_lineage_depth=3,
            pid=os.getpid(),
        )
        with pytest.raises(CoordinationError, match="Timed out waiting for coordination lease"):
            client.acquire_lease(
                instance_id="inst-b",
                lineage_root="lineage-a",
                worker_key="/tmp/project-a::tc=abc::repl=0",
                timeout_seconds=0.2,
            )

        client.release_lease(instance_id="inst-a", lease_id=lease)

        lease_b = client.acquire_lease(
            instance_id="inst-b",
            lineage_root="lineage-a",
            worker_key="/tmp/project-a::tc=abc::repl=0",
            timeout_seconds=0.5,
        )
        assert lease_b
        client.release_lease(instance_id="inst-b", lease_id=lease_b)
        client.unregister_instance(instance_id="inst-a")
        client.unregister_instance(instance_id="inst-b")
    finally:
        # Give the broker a moment to flush last response before shutdown.
        time.sleep(0.02)
        try:
            client.shutdown()
        except Exception:
            pass


@pytest.mark.skipif(os.name == "nt", reason="broker mode is Unix only")
def test_coordination_broker_honors_max_workers_with_same_lineage(
    tmp_path: Path,
) -> None:
    coordination_dir = tmp_path / "coord"
    client = CoordinationClient(coordination_dir=coordination_dir, max_workers=2)

    client.ensure_available(timeout_seconds=5.0)
    try:
        client.register_instance(
            instance_id="inst-a",
            lineage_root="lineage-a",
            lineage_depth=0,
            max_lineage_depth=3,
            pid=os.getpid(),
        )
        client.register_instance(
            instance_id="inst-b",
            lineage_root="lineage-a",
            lineage_depth=0,
            max_lineage_depth=3,
            pid=os.getpid(),
        )

        lease_a = client.acquire_lease(
            instance_id="inst-a",
            lineage_root="lineage-a",
            worker_key="/tmp/project-a::tc=abc::repl=0",
            timeout_seconds=0.2,
        )
        lease_b = client.acquire_lease(
            instance_id="inst-b",
            lineage_root="lineage-a",
            worker_key="/tmp/project-b::tc=def::repl=0",
            timeout_seconds=0.2,
        )
        assert lease_a
        assert lease_b

        client.release_lease(instance_id="inst-a", lease_id=lease_a)
        client.release_lease(instance_id="inst-b", lease_id=lease_b)
        client.unregister_instance(instance_id="inst-a")
        client.unregister_instance(instance_id="inst-b")
    finally:
        # Give the broker a moment to flush last response before shutdown.
        time.sleep(0.02)
        try:
            client.shutdown()
        except Exception:
            pass


@pytest.mark.skipif(os.name == "nt", reason="broker mode is Unix only")
def test_coordination_broker_rejects_negative_lineage_depth(tmp_path: Path) -> None:
    coordination_dir = tmp_path / "coord"
    client = CoordinationClient(coordination_dir=coordination_dir, max_workers=1)

    client.ensure_available(timeout_seconds=5.0)
    try:
        with pytest.raises(CoordinationError, match="must be >= 0"):
            client.register_instance(
                instance_id="inst-neg",
                lineage_root="lineage-a",
                lineage_depth=-1,
                max_lineage_depth=3,
                pid=os.getpid(),
            )
    finally:
        # Give the broker a moment to flush last response before shutdown.
        time.sleep(0.02)
        try:
            client.shutdown()
        except Exception:
            pass


def test_ensure_available_does_not_respawn_on_transient_ping_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = CoordinationClient(coordination_dir=tmp_path / "coord", max_workers=1)
    spawn_calls = 0

    def _request_fail(_payload: dict[str, object], timeout_seconds: float = 2.0):
        _ = timeout_seconds
        raise CoordinationError("Coordination broker connection failed: timed out")

    def _spawn() -> None:
        nonlocal spawn_calls
        spawn_calls += 1

    monkeypatch.setattr(client, "_request", _request_fail)
    monkeypatch.setattr(client, "_spawn_broker", _spawn)

    with pytest.raises(CoordinationError, match="timed out"):
        client.ensure_available(timeout_seconds=0.2)
    assert spawn_calls == 0


def test_ensure_available_respawns_only_for_unreachable_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = CoordinationClient(coordination_dir=tmp_path / "coord", max_workers=1)
    spawn_calls = 0
    ping_calls = 0

    def _request_ping(
        payload: dict[str, object], timeout_seconds: float = 2.0
    ) -> dict[str, object]:
        _ = timeout_seconds
        action = str(payload.get("action"))
        if action == "ping":
            nonlocal ping_calls
            ping_calls += 1
            if ping_calls == 1:
                try:
                    raise OSError(errno.ECONNREFUSED, "Connection refused")
                except OSError as cause:
                    raise CoordinationError(
                        "Coordination broker connection failed: [Errno 61] Connection refused"
                    ) from cause
            return {"pong": True}
        if action == "status":
            return {
                "instances": 0,
                "leases": 0,
                "max_workers": 1,
                "lineage_worker_limit": 1,
            }
        raise AssertionError(f"unexpected action: {action!r}")

    def _spawn() -> None:
        nonlocal spawn_calls
        spawn_calls += 1

    monkeypatch.setattr(client, "_request", _request_ping)
    monkeypatch.setattr(client, "_spawn_broker", _spawn)

    client.ensure_available(timeout_seconds=0.2)
    assert spawn_calls == 1
    assert ping_calls == 2


def test_ensure_available_does_not_unlink_socket_before_respawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = CoordinationClient(coordination_dir=tmp_path / "coord", max_workers=1)
    client.socket_path.parent.mkdir(parents=True, exist_ok=True)
    client.socket_path.write_text("sentinel", encoding="utf-8")
    spawn_calls = 0
    ping_calls = 0

    def _request_ping(
        payload: dict[str, object], timeout_seconds: float = 2.0
    ) -> dict[str, object]:
        _ = timeout_seconds
        action = str(payload.get("action"))
        if action == "ping":
            nonlocal ping_calls
            ping_calls += 1
            if ping_calls == 1:
                try:
                    raise OSError(errno.ECONNREFUSED, "Connection refused")
                except OSError as cause:
                    raise CoordinationError(
                        "Coordination broker connection failed: [Errno 61] Connection refused"
                    ) from cause
            return {"pong": True}
        if action == "status":
            return {
                "instances": 0,
                "leases": 0,
                "max_workers": 1,
                "lineage_worker_limit": 1,
            }
        raise AssertionError(f"unexpected action: {action!r}")

    def _spawn() -> None:
        nonlocal spawn_calls
        spawn_calls += 1

    monkeypatch.setattr(client, "_request", _request_ping)
    monkeypatch.setattr(client, "_spawn_broker", _spawn)

    client.ensure_available(timeout_seconds=0.2)
    assert spawn_calls == 1
    assert ping_calls == 2
    assert client.socket_path.exists()
    assert client.socket_path.read_text(encoding="utf-8") == "sentinel"


def test_ensure_available_rejects_live_broker_limit_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = CoordinationClient(coordination_dir=tmp_path / "coord", max_workers=1)
    spawn_calls = 0

    def _request_status_mismatch(
        payload: dict[str, object], timeout_seconds: float = 2.0
    ) -> dict[str, object]:
        _ = timeout_seconds
        action = str(payload.get("action"))
        if action == "ping":
            return {"pong": True}
        if action == "status":
            return {
                "instances": 0,
                "leases": 0,
                "max_workers": 2,
                "lineage_worker_limit": 1,
            }
        raise AssertionError(f"unexpected action: {action!r}")

    def _spawn() -> None:
        nonlocal spawn_calls
        spawn_calls += 1

    monkeypatch.setattr(client, "_request", _request_status_mismatch)
    monkeypatch.setattr(client, "_spawn_broker", _spawn)

    with pytest.raises(CoordinationError, match="configuration mismatch"):
        client.ensure_available(timeout_seconds=0.2)
    assert spawn_calls == 0


def test_run_broker_process_does_not_unlink_existing_socket_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = tmp_path / "broker.sock"
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.write_text("sentinel", encoding="utf-8")

    class _ExpectExistingPathServer:
        def __init__(self, address: str, _handler: object) -> None:
            if not Path(address).exists():
                raise AssertionError("socket path was removed before bind")
            raise OSError(errno.EADDRINUSE, "Address already in use")

    monkeypatch.setattr(
        "lean_lsp_mcp.coordination._ThreadingUnixStreamServer",
        _ExpectExistingPathServer,
    )

    with pytest.raises(OSError):
        _run_broker_process(socket_path, max_workers=1, lineage_worker_limit=1)

    assert socket_path.exists()
    assert socket_path.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.skipif(os.name == "nt", reason="broker mode is Unix only")
def test_run_broker_process_recovers_stale_socket_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = broker_socket_path(tmp_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a stale socket inode with no listening broker.
    stale_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_socket.bind(str(socket_path))
    stale_socket.close()

    init_calls = 0

    class _RetryServer:
        def __init__(self, address: str, _handler: object) -> None:
            nonlocal init_calls
            init_calls += 1
            if init_calls == 1:
                raise OSError(errno.EADDRINUSE, "Address already in use")
            if Path(address).exists():
                raise AssertionError("stale socket path was not removed before retry bind")
            raise RuntimeError("retry bind attempted")

    monkeypatch.setattr(
        "lean_lsp_mcp.coordination._ThreadingUnixStreamServer",
        _RetryServer,
    )
    monkeypatch.setattr(
        "lean_lsp_mcp.coordination._probe_broker_socket_liveness",
        lambda _path, timeout_seconds=0.2: False,
    )

    with pytest.raises(RuntimeError, match="retry bind attempted"):
        _run_broker_process(socket_path, max_workers=1, lineage_worker_limit=1)

    assert init_calls == 2

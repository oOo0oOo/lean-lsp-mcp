from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from threading import Barrier, Lock as ThreadLock

import pytest

from lean_lsp_mcp.client_utils import (
    CLIENT_LOCK,
    close_client,
    setup_client_for_file,
    startup_client,
    valid_lean_project_path,
)
from lean_lsp_mcp.coordination import CoordinationError
from lean_lsp_mcp.utils import LeanToolError


class _MockLeanClient:
    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FailingCloseClient(_MockLeanClient):
    def __init__(self, project_path: Path) -> None:
        super().__init__(project_path)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise PermissionError("operation not permitted")


class _LifespanContext:
    def __init__(
        self, lean_project_path: Path | None, client: _MockLeanClient | None
    ) -> None:
        self.lean_project_path = lean_project_path
        self.client = client
        self.file_content_hashes: dict[str, int] = {}
        self.coordination_mode = "direct"
        self.coordination_client = None
        self.instance_id = ""
        self.lineage_root = ""
        self.repl_enabled = False
        self.client_lease_id = None
        self.client_worker_key = None


class _RequestContext:
    def __init__(self, lifespan_context: _LifespanContext) -> None:
        self.lifespan_context = lifespan_context


class _Context:
    def __init__(self, lifespan_context: _LifespanContext) -> None:
        self.request_context = _RequestContext(lifespan_context)


class _MockCoordinationClient:
    def __init__(self) -> None:
        self.acquire_calls: list[tuple[str, str, str]] = []
        self.release_calls: list[tuple[str, str]] = []
        self.next_lease = "lease-1"

    def acquire_lease(
        self,
        *,
        instance_id: str,
        lineage_root: str,
        worker_key: str,
        timeout_seconds: float = 10.0,
    ) -> str:
        _ = timeout_seconds
        self.acquire_calls.append((instance_id, lineage_root, worker_key))
        return self.next_lease

    def release_lease(self, *, instance_id: str, lease_id: str) -> None:
        self.release_calls.append((instance_id, lease_id))


class _FailingReleaseCoordinationClient(_MockCoordinationClient):
    def __init__(self, failures_before_success: int) -> None:
        super().__init__()
        self._remaining_failures = failures_before_success

    def release_lease(self, *, instance_id: str, lease_id: str) -> None:
        super().release_lease(instance_id=instance_id, lease_id=lease_id)
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("release failed")


class _LockProbeCoordinationClient(_MockCoordinationClient):
    def __init__(self) -> None:
        super().__init__()
        self.acquire_ran_without_client_lock: bool | None = None

    def acquire_lease(
        self,
        *,
        instance_id: str,
        lineage_root: str,
        worker_key: str,
        timeout_seconds: float = 10.0,
    ) -> str:
        _ = timeout_seconds
        lock_free = CLIENT_LOCK.acquire(blocking=False)
        self.acquire_ran_without_client_lock = lock_free
        if lock_free:
            CLIENT_LOCK.release()
        return super().acquire_lease(
            instance_id=instance_id,
            lineage_root=lineage_root,
            worker_key=worker_key,
            timeout_seconds=timeout_seconds,
        )


class _AcquireFailCoordinationClient(_MockCoordinationClient):
    def acquire_lease(
        self,
        *,
        instance_id: str,
        lineage_root: str,
        worker_key: str,
        timeout_seconds: float = 10.0,
    ) -> str:
        _ = (instance_id, lineage_root, worker_key, timeout_seconds)
        raise CoordinationError("broker unavailable")


class _MutatingAcquireFailingReleaseCoordinationClient(_MockCoordinationClient):
    def __init__(self, *, lifespan: _LifespanContext, new_project_path: Path) -> None:
        super().__init__()
        self._lifespan = lifespan
        self._new_project_path = new_project_path

    def acquire_lease(
        self,
        *,
        instance_id: str,
        lineage_root: str,
        worker_key: str,
        timeout_seconds: float = 10.0,
    ) -> str:
        lease_id = super().acquire_lease(
            instance_id=instance_id,
            lineage_root=lineage_root,
            worker_key=worker_key,
            timeout_seconds=timeout_seconds,
        )
        self._lifespan.lean_project_path = self._new_project_path
        return lease_id

    def release_lease(self, *, instance_id: str, lease_id: str) -> None:
        super().release_lease(instance_id=instance_id, lease_id=lease_id)
        raise RuntimeError("release failed")


class _ConcurrentAcquireCoordinationClient(_MockCoordinationClient):
    def __init__(self) -> None:
        super().__init__()
        self._barrier = Barrier(2)
        self._count_lock = ThreadLock()
        self._acquire_count = 0

    def acquire_lease(
        self,
        *,
        instance_id: str,
        lineage_root: str,
        worker_key: str,
        timeout_seconds: float = 10.0,
    ) -> str:
        _ = timeout_seconds
        with self._count_lock:
            self._acquire_count += 1
            acquire_count = self._acquire_count
        if acquire_count <= 2:
            self._barrier.wait(timeout=2.0)
        return super().acquire_lease(
            instance_id=instance_id,
            lineage_root=lineage_root,
            worker_key=worker_key,
            timeout_seconds=timeout_seconds,
        )


def _worker_key_for(project_path: Path, repl_enabled: bool = False) -> str:
    toolchain_path = project_path / "lean-toolchain"
    toolchain_text = ""
    if toolchain_path.exists():
        toolchain_text = toolchain_path.read_text(encoding="utf-8").strip()
    toolchain_hash = sha256(toolchain_text.encode("utf-8")).hexdigest()[:16]
    repl_flag = "1" if repl_enabled else "0"
    return f"{project_path.resolve()}::tc={toolchain_hash}::repl={repl_flag}"


@pytest.fixture
def patched_clients(monkeypatch: pytest.MonkeyPatch) -> list[_MockLeanClient]:
    created: list[_MockLeanClient] = []

    def _constructor(
        project_path: Path, initial_build: bool, prevent_cache_get: bool = False
    ) -> _MockLeanClient:  # pragma: no cover - signature verified indirectly
        client = _MockLeanClient(project_path)
        created.append(client)
        return client

    monkeypatch.setattr("lean_lsp_mcp.client_utils.LeanLSPClient", _constructor)
    return created


def test_startup_client_reuses_existing(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    ctx = _Context(_LifespanContext(project, None))

    startup_client(ctx)
    first = ctx.request_context.lifespan_context.client
    assert isinstance(first, _MockLeanClient)
    assert not first.closed

    # second startup with same project path should reuse existing client
    startup_client(ctx)
    assert not first.closed

    # change project path triggers close and replacement
    new_project = tmp_path / "proj2"
    new_project.mkdir()
    (new_project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    ctx.request_context.lifespan_context.lean_project_path = new_project

    startup_client(ctx)
    assert first.closed
    assert ctx.request_context.lifespan_context.client.project_path == new_project
    assert len(patched_clients) == 2


def test_startup_client_fails_closed_if_close_fails_during_project_switch(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    old_project = tmp_path / "proj1"
    old_project.mkdir()
    (old_project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    new_project = tmp_path / "proj2"
    new_project.mkdir()
    (new_project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    old_client = _FailingCloseClient(old_project)
    lifespan = _LifespanContext(new_project, old_client)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _MockCoordinationClient()
    lifespan.instance_id = "inst-1"
    lifespan.lineage_root = "lineage-1"
    lifespan.client_lease_id = "lease-old"
    lifespan.client_worker_key = "worker-old"

    with pytest.raises(
        LeanToolError,
        match="refusing to switch projects while shutdown state is uncertain",
    ):
        startup_client(_Context(lifespan))

    assert old_client.close_calls == 1
    assert lifespan.client is old_client
    assert lifespan.client_lease_id == "lease-old"
    assert lifespan.client_worker_key == "worker-old"
    assert lifespan.coordination_client.release_calls == []
    assert lifespan.coordination_client.acquire_calls == []
    assert len(patched_clients) == 0


def test_close_client_keeps_lease_when_client_close_fails(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    old_client = _FailingCloseClient(project)
    lifespan = _LifespanContext(project, old_client)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _MockCoordinationClient()
    lifespan.instance_id = "inst-1"
    lifespan.client_lease_id = "lease-old"
    lifespan.client_worker_key = "worker-old"

    assert close_client(_Context(lifespan)) is False

    assert old_client.close_calls == 1
    assert lifespan.client is old_client
    assert lifespan.client_lease_id == "lease-old"
    assert lifespan.client_worker_key == "worker-old"
    assert lifespan.coordination_client.release_calls == []


def test_close_client_can_preserve_lease_for_restart(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    old_client = _MockLeanClient(project)
    lifespan = _LifespanContext(project, old_client)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _MockCoordinationClient()
    lifespan.instance_id = "inst-1"
    lifespan.client_lease_id = "lease-old"
    lifespan.client_worker_key = "worker-old"

    assert close_client(_Context(lifespan), release_lease=False) is True

    assert old_client.closed
    assert lifespan.client is None
    assert lifespan.client_lease_id == "lease-old"
    assert lifespan.client_worker_key == "worker-old"
    assert lifespan.coordination_client.release_calls == []


def test_valid_lean_project_path(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0")

    assert valid_lean_project_path(project)
    assert not valid_lean_project_path(project / "missing")


def test_setup_client_for_file_discovers_project(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    lean_file = project / "src" / "Example.lean"
    lean_file.parent.mkdir(parents=True)
    lean_file.write_text("example")

    ctx = _Context(_LifespanContext(None, None))

    rel_path = setup_client_for_file(ctx, str(lean_file))
    assert rel_path == "src/Example.lean"
    assert ctx.request_context.lifespan_context.client.project_path == project
    assert len(patched_clients) == 1


def test_setup_client_for_file_reuses_client_for_same_project(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    """Verify that multiple files in the same project reuse the same client."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    file1 = project / "File1.lean"
    file1.write_text("theorem a : True := by trivial")

    file2 = project / "src" / "File2.lean"
    file2.parent.mkdir(parents=True)
    file2.write_text("theorem b : True := by trivial")

    ctx = _Context(_LifespanContext(None, None))

    # Setup for first file
    rel_path1 = setup_client_for_file(ctx, str(file1))
    assert rel_path1 == "File1.lean"
    first_client = ctx.request_context.lifespan_context.client
    assert len(patched_clients) == 1

    # Setup for second file in same project should reuse client
    rel_path2 = setup_client_for_file(ctx, str(file2))
    assert rel_path2 == "src/File2.lean"
    assert ctx.request_context.lifespan_context.client is first_client
    assert not first_client.closed
    assert len(patched_clients) == 1  # No new client created


def test_setup_client_for_file_switches_projects(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    """Verify that switching to a different project closes old client and creates new one."""
    project1 = tmp_path / "proj1"
    project1.mkdir()
    (project1 / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    file1 = project1 / "File1.lean"
    file1.write_text("theorem a : True := by trivial")

    project2 = tmp_path / "proj2"
    project2.mkdir()
    (project2 / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    file2 = project2 / "File2.lean"
    file2.write_text("theorem b : True := by trivial")

    ctx = _Context(_LifespanContext(None, None))

    # Setup for first project
    rel_path1 = setup_client_for_file(ctx, str(file1))
    assert rel_path1 == "File1.lean"
    first_client = ctx.request_context.lifespan_context.client
    assert len(patched_clients) == 1

    # Switch to second project
    rel_path2 = setup_client_for_file(ctx, str(file2))
    assert rel_path2 == "File2.lean"
    second_client = ctx.request_context.lifespan_context.client

    # Old client should be closed, new one created
    assert first_client.closed
    assert second_client is not first_client
    assert len(patched_clients) == 2
    assert ctx.request_context.lifespan_context.lean_project_path == project2


def test_startup_client_serializes_concurrent_calls(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    ctx = _Context(_LifespanContext(project, None))

    def _invoke_startup() -> None:
        startup_client(ctx)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_invoke_startup) for _ in range(10)]
        for future in futures:
            assert future.result() is None

    assert len(patched_clients) == 1
    assert ctx.request_context.lifespan_context.client is patched_clients[0]


def test_startup_client_acquires_and_releases_broker_lease(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    lifespan = _LifespanContext(project, None)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _MockCoordinationClient()
    lifespan.instance_id = "inst-1"
    lifespan.lineage_root = "lineage-1"
    ctx = _Context(lifespan)

    startup_client(ctx)

    assert len(patched_clients) == 1
    assert lifespan.client is patched_clients[0]
    assert lifespan.client_lease_id == "lease-1"
    assert len(lifespan.coordination_client.acquire_calls) == 1

    # Reusing same project/client should not acquire a second lease.
    startup_client(ctx)
    assert len(lifespan.coordination_client.acquire_calls) == 1

    assert close_client(ctx) is True
    assert lifespan.client is None
    assert lifespan.client_lease_id is None
    assert lifespan.coordination_client.release_calls == [("inst-1", "lease-1")]


def test_startup_client_acquires_lease_outside_global_client_lock(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    lifespan = _LifespanContext(project, None)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _LockProbeCoordinationClient()
    lifespan.instance_id = "inst-1"
    lifespan.lineage_root = "lineage-1"

    startup_client(_Context(lifespan))

    assert len(patched_clients) == 1
    assert lifespan.coordination_client.acquire_ran_without_client_lock is True


def test_startup_client_reacquires_cached_broker_lease_before_start(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    worker_key = _worker_key_for(project)

    lifespan = _LifespanContext(project, None)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _MockCoordinationClient()
    lifespan.coordination_client.next_lease = "lease-new"
    lifespan.instance_id = "inst-1"
    lifespan.lineage_root = "lineage-1"
    lifespan.client_lease_id = "lease-old"
    lifespan.client_worker_key = worker_key

    startup_client(_Context(lifespan))

    assert len(patched_clients) == 1
    assert lifespan.coordination_client.acquire_calls == [
        ("inst-1", "lineage-1", worker_key)
    ]
    assert lifespan.client_lease_id == "lease-new"
    assert lifespan.client_worker_key == worker_key


def test_startup_client_fails_when_reacquire_fails_with_cached_lease(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    worker_key = _worker_key_for(project)

    lifespan = _LifespanContext(project, None)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _AcquireFailCoordinationClient()
    lifespan.instance_id = "inst-1"
    lifespan.lineage_root = "lineage-1"
    lifespan.client_lease_id = "lease-old"
    lifespan.client_worker_key = worker_key

    with pytest.raises(LeanToolError, match="Failed to acquire coordinated Lean worker lease"):
        startup_client(_Context(lifespan))

    assert len(patched_clients) == 0
    assert lifespan.client is None
    assert lifespan.client_lease_id == "lease-old"
    assert lifespan.client_worker_key == worker_key


def test_startup_client_does_not_release_shared_lease_on_parallel_init_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    worker_key = _worker_key_for(project)

    lifespan = _LifespanContext(project, None)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _ConcurrentAcquireCoordinationClient()
    lifespan.instance_id = "inst-1"
    lifespan.lineage_root = "lineage-1"
    ctx = _Context(lifespan)

    created_clients: list[_MockLeanClient] = []
    create_lock = ThreadLock()
    create_calls = 0

    def _constructor(
        project_path: Path, initial_build: bool, prevent_cache_get: bool = False
    ) -> _MockLeanClient:
        nonlocal create_calls
        _ = (initial_build, prevent_cache_get)
        with create_lock:
            create_calls += 1
            call_num = create_calls
        if call_num == 1:
            raise RuntimeError("client init failed")
        client = _MockLeanClient(project_path)
        created_clients.append(client)
        return client

    monkeypatch.setattr("lean_lsp_mcp.client_utils.LeanLSPClient", _constructor)

    failures = 0

    def _invoke_startup() -> None:
        nonlocal failures
        try:
            startup_client(ctx)
        except RuntimeError as exc:
            assert "client init failed" in str(exc)
            with create_lock:
                failures += 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_invoke_startup) for _ in range(2)]
        for future in futures:
            assert future.result() is None

    assert failures == 1
    assert len(created_clients) == 1
    assert lifespan.client is created_clients[0]
    assert lifespan.client_lease_id == "lease-1"
    assert lifespan.client_worker_key == worker_key
    assert lifespan.coordination_client.acquire_calls == [
        ("inst-1", "lineage-1", worker_key),
        ("inst-1", "lineage-1", worker_key),
    ]
    assert lifespan.coordination_client.release_calls == []


def test_startup_client_preserves_lease_on_init_failure_until_explicit_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    worker_key = _worker_key_for(project)

    lifespan = _LifespanContext(project, None)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _MockCoordinationClient()
    lifespan.instance_id = "inst-1"
    lifespan.lineage_root = "lineage-1"
    ctx = _Context(lifespan)

    def _constructor(
        project_path: Path, initial_build: bool, prevent_cache_get: bool = False
    ) -> _MockLeanClient:
        _ = (project_path, initial_build, prevent_cache_get)
        raise RuntimeError("client init failed")

    monkeypatch.setattr("lean_lsp_mcp.client_utils.LeanLSPClient", _constructor)

    with pytest.raises(RuntimeError, match="client init failed"):
        startup_client(ctx)

    assert lifespan.client is None
    assert lifespan.client_lease_id == "lease-1"
    assert lifespan.client_worker_key == worker_key
    assert lifespan.coordination_client.acquire_calls == [
        ("inst-1", "lineage-1", worker_key)
    ]
    assert lifespan.coordination_client.release_calls == []

    assert close_client(ctx) is True
    assert lifespan.client_lease_id is None
    assert lifespan.client_worker_key is None
    assert lifespan.coordination_client.release_calls == [("inst-1", "lease-1")]


def test_startup_client_fails_closed_if_stale_cleanup_release_fails(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    moved_project = tmp_path / "proj-moved"
    moved_project.mkdir()
    (moved_project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    lifespan = _LifespanContext(project, None)
    lifespan.coordination_mode = "broker"
    lifespan.instance_id = "inst-1"
    lifespan.lineage_root = "lineage-1"
    lifespan.coordination_client = _MutatingAcquireFailingReleaseCoordinationClient(
        lifespan=lifespan,
        new_project_path=moved_project,
    )

    with pytest.raises(
        LeanToolError,
        match="Failed to release stale coordinated Lean worker lease after project change",
    ):
        startup_client(_Context(lifespan))

    assert len(patched_clients) == 0
    assert lifespan.client is None
    assert lifespan.coordination_client.acquire_calls == [
        ("inst-1", "lineage-1", _worker_key_for(project))
    ]
    assert lifespan.coordination_client.release_calls == [("inst-1", "lease-1")]


def test_close_client_preserves_lease_metadata_when_release_fails(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    lifespan = _LifespanContext(project, None)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _FailingReleaseCoordinationClient(
        failures_before_success=1
    )
    lifespan.instance_id = "inst-1"
    lifespan.client_lease_id = "lease-1"
    lifespan.client_worker_key = "worker-a"

    assert close_client(_Context(lifespan)) is False

    assert lifespan.client_lease_id == "lease-1"
    assert lifespan.client_worker_key == "worker-a"
    assert lifespan.coordination_client.release_calls == [("inst-1", "lease-1")]


def test_startup_client_retries_stale_lease_release_on_project_switch(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    old_project = tmp_path / "proj1"
    old_project.mkdir()
    (old_project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    new_project = tmp_path / "proj2"
    new_project.mkdir()
    (new_project / "lean-toolchain").write_text("leanprover/lean4:v4.24.1\n")

    old_client = _MockLeanClient(old_project)
    lifespan = _LifespanContext(new_project, old_client)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _FailingReleaseCoordinationClient(
        failures_before_success=1
    )
    lifespan.instance_id = "inst-1"
    lifespan.lineage_root = "lineage-1"
    lifespan.client_lease_id = "lease-old"
    lifespan.client_worker_key = "worker-old"

    startup_client(_Context(lifespan))

    assert old_client.closed
    assert len(patched_clients) == 1
    assert lifespan.client is patched_clients[0]
    assert lifespan.client_lease_id == "lease-1"
    assert lifespan.client_worker_key != "worker-old"
    assert lifespan.coordination_client.acquire_calls == [
        ("inst-1", "lineage-1", lifespan.client_worker_key)
    ]
    assert lifespan.coordination_client.release_calls == [
        ("inst-1", "lease-old"),
        ("inst-1", "lease-old"),
    ]


def test_startup_client_fails_closed_if_stale_lease_cannot_be_released(
    tmp_path: Path, patched_clients: list[_MockLeanClient]
) -> None:
    old_project = tmp_path / "proj1"
    old_project.mkdir()
    (old_project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    new_project = tmp_path / "proj2"
    new_project.mkdir()
    (new_project / "lean-toolchain").write_text("leanprover/lean4:v4.24.1\n")

    old_client = _MockLeanClient(old_project)
    lifespan = _LifespanContext(new_project, old_client)
    lifespan.coordination_mode = "broker"
    lifespan.coordination_client = _FailingReleaseCoordinationClient(
        failures_before_success=5
    )
    lifespan.instance_id = "inst-1"
    lifespan.lineage_root = "lineage-1"
    lifespan.client_lease_id = "lease-old"
    lifespan.client_worker_key = "worker-old"

    with pytest.raises(
        LeanToolError,
        match="cannot switch projects while lease state is uncertain",
    ):
        startup_client(_Context(lifespan))

    assert old_client.closed
    assert lifespan.client is None
    assert lifespan.client_lease_id == "lease-old"
    assert lifespan.client_worker_key == "worker-old"
    assert lifespan.coordination_client.acquire_calls == []
    assert lifespan.coordination_client.release_calls == [
        ("inst-1", "lease-old"),
        ("inst-1", "lease-old"),
    ]

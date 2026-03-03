from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lean_lsp_mcp import server


class DummyClient:
    def __init__(self) -> None:
        self.closed_calls = 0

    def close(self) -> None:
        self.closed_calls += 1


class _FailingCloseClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise PermissionError("operation not permitted")


class _FailingRepl:
    instances: list["_FailingRepl"] = []

    def __init__(self, project_dir: str, repl_path: str) -> None:
        self.timeout = 60
        self.close_calls = 0
        self.project_dir = project_dir
        self.repl_path = repl_path
        self.__class__.instances.append(self)

    async def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("close boom")


class _MockCoordinationClient:
    instances: list["_MockCoordinationClient"] = []

    def __init__(self, **_kwargs) -> None:
        self.ensure_calls = 0
        self.register_calls: list[dict[str, object]] = []
        self.unregister_calls: list[str] = []
        self.release_calls: list[tuple[str, str]] = []
        self.__class__.instances.append(self)

    def ensure_available(self) -> None:
        self.ensure_calls += 1

    def register_instance(self, **kwargs) -> None:
        self.register_calls.append(kwargs)

    def unregister_instance(self, *, instance_id: str) -> None:
        self.unregister_calls.append(instance_id)

    def release_lease(self, *, instance_id: str, lease_id: str) -> None:
        self.release_calls.append((instance_id, lease_id))


def _make_ctx(rate_limit: dict[str, list[int]] | None = None) -> types.SimpleNamespace:
    context = server.AppContext(
        lean_project_path=None,
        client=None,
        rate_limit=rate_limit or {"test": []},
        lean_search_available=True,
    )
    request_context = types.SimpleNamespace(lifespan_context=context)
    return types.SimpleNamespace(request_context=request_context)


@pytest.mark.asyncio
async def test_app_lifespan_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEAN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("LEAN_PROJECT_PATH", raising=False)

    async with server.app_lifespan(object()) as context:
        assert context.lean_project_path is None
        assert context.client is None
        assert context.rate_limit == {
            "leansearch": [],
            "loogle": [],
            "leanfinder": [],
            "lean_state_search": [],
            "hammer_premise": [],
        }


@pytest.mark.asyncio
async def test_app_lifespan_sets_project_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    monkeypatch.setenv("LEAN_PROJECT_PATH", str(project_dir))

    async with server.app_lifespan(object()) as context:
        assert context.lean_project_path == project_dir.resolve()


@pytest.mark.asyncio
async def test_app_lifespan_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEAN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("LEAN_PROJECT_PATH", raising=False)

    dummy_client = DummyClient()

    async with server.app_lifespan(object()) as context:
        context.client = dummy_client

    assert dummy_client.closed_calls == 1


@pytest.mark.asyncio
async def test_app_lifespan_suppresses_client_close_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEAN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("LEAN_PROJECT_PATH", raising=False)

    dummy_client = _FailingCloseClient()

    async with server.app_lifespan(object()) as context:
        context.client = dummy_client

    assert dummy_client.close_calls == 1


@pytest.mark.asyncio
async def test_app_lifespan_keeps_lease_if_client_close_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _MockCoordinationClient.instances.clear()
    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION", "broker")
    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION_DIR", str(tmp_path / "coord"))
    monkeypatch.setenv("LEAN_LSP_MCP_INSTANCE_ID", "inst-123")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_ROOT", "lineage-1")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_DEPTH", "1")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_LINEAGE_DEPTH", "3")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_WORKERS", "2")
    monkeypatch.setattr(server, "CoordinationClient", _MockCoordinationClient)

    failing_client = _FailingCloseClient()
    async with server.app_lifespan(object()) as context:
        context.client = failing_client
        context.client_lease_id = "lease-1"
        context.client_worker_key = "worker-1"

    assert failing_client.close_calls == 1
    assert len(_MockCoordinationClient.instances) == 1
    broker = _MockCoordinationClient.instances[0]
    assert broker.release_calls == []
    assert broker.unregister_calls == []


@pytest.mark.asyncio
async def test_app_lifespan_unregisters_if_client_close_fails_without_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _MockCoordinationClient.instances.clear()
    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION", "broker")
    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION_DIR", str(tmp_path / "coord"))
    monkeypatch.setenv("LEAN_LSP_MCP_INSTANCE_ID", "inst-123")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_ROOT", "lineage-1")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_DEPTH", "1")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_LINEAGE_DEPTH", "3")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_WORKERS", "2")
    monkeypatch.setattr(server, "CoordinationClient", _MockCoordinationClient)

    failing_client = _FailingCloseClient()
    async with server.app_lifespan(object()) as context:
        context.client = failing_client

    assert failing_client.close_calls == 1
    assert len(_MockCoordinationClient.instances) == 1
    broker = _MockCoordinationClient.instances[0]
    registered_id = str(broker.register_calls[0]["instance_id"])
    assert broker.release_calls == []
    assert broker.unregister_calls == [registered_id]


@pytest.mark.asyncio
async def test_app_lifespan_suppresses_repl_close_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _FailingRepl.instances.clear()
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    monkeypatch.setenv("LEAN_PROJECT_PATH", str(project_dir))
    monkeypatch.setattr(server, "repl_enabled", lambda: True)
    monkeypatch.setattr(
        "lean_lsp_mcp.repl.find_repl_binary",
        lambda _path: "/tmp/repl",
    )
    monkeypatch.setattr(server, "Repl", _FailingRepl)

    async with server.app_lifespan(object()) as context:
        assert context.repl is not None

    assert len(_FailingRepl.instances) == 1
    assert _FailingRepl.instances[0].close_calls == 1


@pytest.mark.asyncio
async def test_app_lifespan_preserves_startup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(_project_path):
        raise RuntimeError("startup boom")

    monkeypatch.setattr(server, "_ensure_shared_loogle", boom)

    with pytest.raises(RuntimeError, match="startup boom"):
        async with server.app_lifespan(object()):
            pass


@pytest.mark.asyncio
async def test_app_lifespan_unregisters_broker_on_startup_failure_after_register(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def boom(_project_path):
        raise RuntimeError("startup boom")

    _MockCoordinationClient.instances.clear()
    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION", "broker")
    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION_DIR", str(tmp_path / "coord"))
    monkeypatch.setenv("LEAN_LSP_MCP_INSTANCE_ID", "inst-123")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_ROOT", "lineage-1")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_DEPTH", "1")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_LINEAGE_DEPTH", "3")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_WORKERS", "2")
    monkeypatch.setattr(server, "CoordinationClient", _MockCoordinationClient)
    monkeypatch.setattr(server, "_ensure_shared_loogle", boom)

    with pytest.raises(RuntimeError, match="startup boom"):
        async with server.app_lifespan(object()):
            pass

    assert len(_MockCoordinationClient.instances) == 1
    broker = _MockCoordinationClient.instances[0]
    assert len(broker.register_calls) == 1
    registered_id = str(broker.register_calls[0]["instance_id"])
    assert broker.unregister_calls == [registered_id]


@pytest.mark.asyncio
async def test_app_lifespan_broker_registers_and_unregisters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _MockCoordinationClient.instances.clear()
    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION", "broker")
    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION_DIR", str(tmp_path / "coord"))
    monkeypatch.setenv("LEAN_LSP_MCP_INSTANCE_ID", "inst-123")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_ROOT", "lineage-1")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_DEPTH", "1")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_LINEAGE_DEPTH", "3")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_WORKERS", "2")
    monkeypatch.setattr(server, "CoordinationClient", _MockCoordinationClient)

    async with server.app_lifespan(object()) as context:
        assert context.coordination_client is not None
        assert context.instance_id.startswith("inst-123:")
        assert context.lineage_root == "lineage-1"
        assert context.lineage_depth == 1

    assert len(_MockCoordinationClient.instances) == 1
    client = _MockCoordinationClient.instances[0]
    assert client.ensure_calls == 1
    registered_id = str(client.register_calls[0]["instance_id"])
    assert registered_id.startswith("inst-123:")
    assert client.unregister_calls == [registered_id]


@pytest.mark.asyncio
async def test_app_lifespan_broker_uses_unique_instance_id_per_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _MockCoordinationClient.instances.clear()
    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION", "broker")
    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION_DIR", str(tmp_path / "coord"))
    monkeypatch.setenv("LEAN_LSP_MCP_INSTANCE_ID", "inst-123")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_ROOT", "lineage-1")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_DEPTH", "1")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_LINEAGE_DEPTH", "3")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_WORKERS", "2")
    monkeypatch.setattr(server, "CoordinationClient", _MockCoordinationClient)

    async with server.app_lifespan(object()) as context_a:
        id_a = context_a.instance_id
    async with server.app_lifespan(object()) as context_b:
        id_b = context_b.instance_id

    assert id_a != id_b
    assert id_a.startswith("inst-123:")
    assert id_b.startswith("inst-123:")


@pytest.mark.asyncio
async def test_app_lifespan_broker_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _FailingBroker:
        def __init__(self, **_kwargs) -> None:
            return

        def ensure_available(self) -> None:
            raise server.CoordinationError("broker down")

    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION", "broker")
    monkeypatch.setenv("LEAN_LSP_MCP_COORDINATION_DIR", str(tmp_path / "coord"))
    monkeypatch.setenv("LEAN_LSP_MCP_INSTANCE_ID", "inst-123")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_ROOT", "lineage-1")
    monkeypatch.setenv("LEAN_LSP_MCP_LINEAGE_DEPTH", "1")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_LINEAGE_DEPTH", "3")
    monkeypatch.setenv("LEAN_LSP_MCP_MAX_WORKERS", "2")
    monkeypatch.setattr(server, "CoordinationClient", _FailingBroker)

    with pytest.raises(
        RuntimeError, match="Broker coordination startup failed \\(fail-closed\\)"
    ):
        async with server.app_lifespan(object()):
            pass


def test_rate_limited_allows_within_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([100, 101])
    monkeypatch.setattr(server.time, "time", lambda: next(times))

    @server.rate_limited("test", max_requests=2, per_seconds=10)
    def wrapped(*, ctx: types.SimpleNamespace) -> str:
        """Test helper"""
        return "ok"

    ctx = _make_ctx()
    assert wrapped(ctx=ctx) == "ok"
    assert wrapped(ctx=ctx) == "ok"


def test_rate_limited_blocks_excess(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([100, 101, 102])
    monkeypatch.setattr(server.time, "time", lambda: next(times))

    @server.rate_limited("test", max_requests=2, per_seconds=10)
    def wrapped(*, ctx: types.SimpleNamespace) -> str:
        """Test helper"""
        return "ok"

    ctx = _make_ctx()
    assert wrapped(ctx=ctx) == "ok"
    assert wrapped(ctx=ctx) == "ok"
    message = wrapped(ctx=ctx)
    assert "Tool limit exceeded" in message


def test_rate_limited_trims_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([100])
    monkeypatch.setattr(server.time, "time", lambda: next(times))

    rate_limit = {"test": [80, 81]}
    ctx = _make_ctx(rate_limit=rate_limit)

    @server.rate_limited("test", max_requests=2, per_seconds=10)
    def wrapped(*, ctx: types.SimpleNamespace) -> str:
        """Test helper"""
        return "ok"

    assert wrapped(ctx=ctx) == "ok"
    assert rate_limit["test"] == [100]


@pytest.mark.asyncio
async def test_local_search_project_root_updates_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_result = [{"name": "foo", "kind": "def", "file": "Foo.lean"}]
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    called: dict[str, Path] = {}

    def fake_search(*, query: str, limit: int, project_root: Path):
        called["query"] = query
        called["limit"] = limit
        called["root"] = project_root
        return fake_result

    monkeypatch.setattr(server, "_RG_AVAILABLE", True)
    monkeypatch.setattr(server, "lean_local_search", fake_search)

    ctx = _make_ctx()

    result = await server.local_search(
        ctx=ctx, query=" foo ", limit=7, project_root=str(project_dir)
    )

    # Result is now a LocalSearchResults model with items field
    assert len(result.items) == 1
    assert result.items[0].name == "foo"
    assert result.items[0].kind == "def"
    assert result.items[0].file == "Foo.lean"
    assert called == {
        "query": "foo",
        "limit": 7,
        "root": project_dir.resolve(),
    }
    assert (
        ctx.request_context.lifespan_context.lean_project_path == project_dir.resolve()
    )


@pytest.mark.asyncio
async def test_local_search_requires_project_root_when_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(server, "_RG_AVAILABLE", True)

    ctx = _make_ctx()
    missing_path = tmp_path / "missing"

    # Now raises LocalSearchError instead of returning error string
    with pytest.raises(server.LocalSearchError) as exc_info:
        await server.local_search(ctx=ctx, query="foo", project_root=str(missing_path))

    assert "does not exist" in str(exc_info.value)


@pytest.mark.asyncio
async def test_shared_loogle_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two calls to _ensure_shared_loogle return the same manager instance."""
    # Reset shared state
    server._shared_loogle_init_done = False
    server._shared_loogle_manager = None
    server._shared_loogle_available = False

    monkeypatch.setenv("LEAN_LOOGLE_LOCAL", "true")

    fake_manager = MagicMock()
    fake_manager.ensure_installed.return_value = True
    fake_manager.start = AsyncMock(return_value=True)

    monkeypatch.setattr(server, "LoogleManager", lambda **_kwargs: fake_manager)

    mgr1, avail1 = await server._ensure_shared_loogle(None)
    mgr2, avail2 = await server._ensure_shared_loogle(None)

    assert mgr1 is mgr2
    assert mgr1 is fake_manager
    assert avail1 is True
    assert avail2 is True
    # LoogleManager constructed and started only once
    assert fake_manager.ensure_installed.call_count == 1
    assert fake_manager.start.call_count == 1


@pytest.mark.asyncio
async def test_shared_loogle_retries_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server._shared_loogle_init_done = False
    server._shared_loogle_manager = None
    server._shared_loogle_available = False

    monkeypatch.setenv("LEAN_LOOGLE_LOCAL", "true")

    fake_manager = MagicMock()
    fake_manager.ensure_installed.return_value = True
    fake_manager.start = AsyncMock(side_effect=[False, True])

    monkeypatch.setattr(server, "LoogleManager", lambda **_kwargs: fake_manager)

    mgr1, avail1 = await server._ensure_shared_loogle(None)
    assert mgr1 is fake_manager
    assert avail1 is False
    assert server._shared_loogle_init_done is False

    mgr2, avail2 = await server._ensure_shared_loogle(None)
    assert mgr2 is fake_manager
    assert avail2 is True
    assert server._shared_loogle_init_done is True
    assert fake_manager.start.call_count == 2


class _BaseMultiAttemptClient:
    def __init__(self) -> None:
        self.open_calls: list[tuple[str, bool]] = []
        self.restore_calls: list[tuple[str, str]] = []

    def open_file(
        self,
        path: str,
        dependency_build_mode: str = "never",
        force_reopen: bool = False,
    ) -> None:
        _ = dependency_build_mode
        self.open_calls.append((path, force_reopen))

    def update_file(self, _path: str, _changes: list[object]) -> None:
        return

    def get_diagnostics(self, _path: str) -> list[dict]:
        return []

    def get_goal(self, _path: str, _line: int, _column: int) -> dict:
        return {}

    def update_file_content(self, path: str, content: str) -> None:
        self.restore_calls.append((path, content))


def test_multi_attempt_force_reopens_after_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient(_BaseMultiAttemptClient):
        def get_file_content(self, _path: str) -> str:
            return "buffer-content"

    fake_client = FakeClient()
    ctx = _make_ctx()
    ctx.request_context.lifespan_context.client = fake_client

    monkeypatch.setattr(server, "setup_client_for_file", lambda _ctx, _path: "Foo.lean")
    monkeypatch.setattr(server, "get_file_contents", lambda _path: "original")

    result = server._multi_attempt_lsp(ctx, "/abs/Foo.lean", line=1, snippets=[])

    assert result.items == []
    assert fake_client.restore_calls == [("Foo.lean", "buffer-content")]
    assert fake_client.open_calls == [("Foo.lean", False), ("Foo.lean", True)]


def test_multi_attempt_restore_falls_back_to_disk_on_buffer_read_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient(_BaseMultiAttemptClient):
        def get_file_content(self, _path: str) -> str:
            raise RuntimeError("buffer unavailable")

    fake_client = FakeClient()
    ctx = _make_ctx()
    ctx.request_context.lifespan_context.client = fake_client

    monkeypatch.setattr(server, "setup_client_for_file", lambda _ctx, _path: "Foo.lean")
    monkeypatch.setattr(server, "get_file_contents", lambda _path: "disk-content")

    result = server._multi_attempt_lsp(ctx, "/abs/Foo.lean", line=1, snippets=[])

    assert result.items == []
    assert fake_client.restore_calls == [("Foo.lean", "disk-content")]

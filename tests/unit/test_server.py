from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError
import importlib
import json
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lean_lsp_mcp import client_utils
from lean_lsp_mcp import server
from lean_lsp_mcp.models import DiagnosticSeverity


class _FakeTransport:
    def __init__(self) -> None:
        self.kill_calls = 0

    def _kill_group(self) -> None:
        self.kill_calls += 1


class DummyClient:
    def __init__(self) -> None:
        self._transport = _FakeTransport()


def _async_setup(rel_path):
    async def fake_setup(_ctx, _path):
        return rel_path

    return fake_setup


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


@pytest.fixture(autouse=True)
def _clear_optional_runtime_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEAN_LSP_MCP_ACTIVE_TRANSPORT", raising=False)
    monkeypatch.delenv("LEAN_PROJECT_PATH", raising=False)
    monkeypatch.delenv("LEAN_LSP_MCP_TOKEN", raising=False)
    monkeypatch.delenv("LEAN_LOOGLE_LOCAL", raising=False)
    monkeypatch.delenv("LEAN_REPL", raising=False)
    monkeypatch.delenv("LEAN_LSP_MCP_ACTIVE_TRANSPORT", raising=False)
    client_utils.close_shared_client()


def _make_ctx(
    rate_limit: dict[str, list[int]] | None = None,
    *,
    lean_project_path: Path | None = None,
    active_transport: str = "stdio",
    project_switching_allowed: bool = True,
) -> types.SimpleNamespace:
    context = server.AppContext(
        lean_project_path=lean_project_path,
        client=None,
        rate_limit=rate_limit or {"test": []},
        lean_search_available=True,
        active_transport=active_transport,
        project_switching_allowed=project_switching_allowed,
    )
    request_context = types.SimpleNamespace(lifespan_context=context)
    return types.SimpleNamespace(request_context=request_context)


def _make_project(root: Path) -> Path:
    root.mkdir()
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    (root / "lakefile.toml").write_text('name = "test"\n')
    return root


def _make_dependency(project: Path, dep_root: Path) -> Path:
    dep_file = dep_root / "Mathlib" / "Foo.lean"
    dep_file.parent.mkdir(parents=True)
    dep_file.write_text("theorem dep : True := by trivial\n")

    dep_link = project / ".lake" / "packages" / "mathlib"
    dep_link.parent.mkdir(parents=True)
    dep_link.symlink_to(dep_root, target_is_directory=True)
    return dep_file


@pytest.mark.asyncio
async def test_app_lifespan_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEAN_LOG_LEVEL", raising=False)

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
    project_dir = _make_project(tmp_path / "proj")
    monkeypatch.setenv("LEAN_PROJECT_PATH", str(project_dir))

    async with server.app_lifespan(object()) as context:
        assert context.lean_project_path == project_dir.resolve()


@pytest.mark.asyncio
async def test_app_lifespan_requires_project_path_for_remote_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEAN_PROJECT_PATH", raising=False)
    monkeypatch.setenv("LEAN_LSP_MCP_ACTIVE_TRANSPORT", "streamable-http")

    with pytest.raises(ValueError, match="LEAN_PROJECT_PATH"):
        async with server.app_lifespan(object()):
            pass


@pytest.mark.asyncio
async def test_app_lifespan_does_not_close_shared_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The LSP client is a shared singleton — app_lifespan must NOT close it."""
    monkeypatch.delenv("LEAN_LOG_LEVEL", raising=False)

    dummy_client = DummyClient()

    async with server.app_lifespan(object()) as context:
        context.client = dummy_client

    assert dummy_client._transport.kill_calls == 0


def test_close_shared_client_closes_client() -> None:
    """close_shared_client() kills the shared singleton's process group."""
    dummy = DummyClient()
    client_utils._shared_clients[Path("/tmp/proj")] = dummy

    try:
        client_utils.close_shared_client()
        assert dummy._transport.kill_calls == 1
        assert client_utils._shared_clients == {}
    finally:
        client_utils._shared_clients.clear()


def test_close_shared_client_suppresses_error() -> None:
    """close_shared_client() suppresses exceptions from the terminate path."""
    dummy = DummyClient()

    def _boom() -> None:
        raise PermissionError("operation not permitted")

    dummy._transport._kill_group = _boom
    client_utils._shared_clients[Path("/tmp/proj")] = dummy

    try:
        client_utils.close_shared_client()  # should not raise
        assert client_utils._shared_clients == {}
    finally:
        client_utils._shared_clients.clear()


def test_close_shared_client_noop_when_none() -> None:
    """close_shared_client() is safe to call when no client exists."""
    client_utils._shared_clients.clear()
    client_utils.close_shared_client()  # should not raise


@pytest.mark.asyncio
async def test_app_lifespan_suppresses_repl_close_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _FailingRepl.instances.clear()
    project_dir = _make_project(tmp_path / "proj")
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
async def test_app_lifespan_disables_repl_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = _make_project(tmp_path / "proj")
    monkeypatch.setenv("LEAN_PROJECT_PATH", str(project_dir))
    monkeypatch.setattr(server, "repl_enabled", lambda: True)
    monkeypatch.setattr(
        "lean_lsp_mcp.repl.find_repl_binary",
        lambda _path: None,
    )

    async with server.app_lifespan(object()) as context:
        assert context.repl is None
        assert context.repl_enabled is False


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


def test_server_configures_required_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEAN_LSP_MCP_TOKEN", "secret")

    try:
        importlib.reload(server)

        assert server.mcp.settings.auth is not None
        verifier = server.mcp._token_verifier
        assert verifier is not None
        assert asyncio.run(verifier.verify_token("secret")) is not None
        assert asyncio.run(verifier.verify_token("wrong")) is None
    finally:
        monkeypatch.delenv("LEAN_LSP_MCP_TOKEN", raising=False)
        importlib.reload(server)


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
    with pytest.raises(server.LeanToolError, match="Tool limit exceeded"):
        wrapped(ctx=ctx)


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


def test_rate_limited_bypass_skips_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """When bypass() is true (custom backend), the limit is not applied."""
    times = iter([100, 101, 102])
    monkeypatch.setattr(server.time, "time", lambda: next(times))

    @server.rate_limited("test", max_requests=1, per_seconds=10, bypass=lambda: True)
    def wrapped(*, ctx: types.SimpleNamespace) -> str:
        """Test helper"""
        return "ok"

    ctx = _make_ctx()
    assert wrapped(ctx=ctx) == "ok"
    assert wrapped(ctx=ctx) == "ok"
    assert wrapped(ctx=ctx) == "ok"


def test_custom_backend_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    default = "https://premise-search.com"

    monkeypatch.delenv("LEAN_STATE_SEARCH_URL", raising=False)
    assert server._custom_backend("LEAN_STATE_SEARCH_URL", default) is False

    monkeypatch.setenv("LEAN_STATE_SEARCH_URL", default)
    assert server._custom_backend("LEAN_STATE_SEARCH_URL", default) is False

    # Trailing-slash difference should still count as the default.
    monkeypatch.setenv("LEAN_STATE_SEARCH_URL", default + "/")
    assert server._custom_backend("LEAN_STATE_SEARCH_URL", default) is False

    monkeypatch.setenv("LEAN_STATE_SEARCH_URL", "http://localhost:8000")
    assert server._custom_backend("LEAN_STATE_SEARCH_URL", default) is True


def test_parse_disabled_tools() -> None:
    assert server._parse_disabled_tools(None) == set()
    assert server._parse_disabled_tools("") == set()
    assert server._parse_disabled_tools("lean_build, lean_run_code ,,") == {
        "lean_build",
        "lean_run_code",
    }


def test_load_tool_description_overrides_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LEAN_MCP_TOOL_DESCRIPTIONS",
        json.dumps(
            {
                "lean_build": "Build tool from env",
                "lean_goal": "Goal tool from env",
            }
        ),
    )

    overrides = server._load_tool_description_overrides()
    assert overrides["lean_build"] == "Build tool from env"
    assert overrides["lean_goal"] == "Goal tool from env"


def test_apply_tool_configuration_disables_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp = server.FastMCP(name="test", instructions="base instructions")

    @mcp.tool("enabled_tool")
    def enabled_tool() -> str:
        """enabled description"""
        return "ok"

    @mcp.tool("removed_tool")
    def removed_tool() -> str:
        """removed description"""
        return "ok"

    monkeypatch.setenv("LEAN_MCP_DISABLED_TOOLS", "removed_tool")
    monkeypatch.setenv("LEAN_MCP_INSTRUCTIONS", "custom server instructions")
    monkeypatch.setenv(
        "LEAN_MCP_TOOL_DESCRIPTIONS",
        json.dumps({"enabled_tool": "overridden description"}),
    )

    server.apply_tool_configuration(mcp)

    assert mcp.instructions == "custom server instructions"
    assert mcp._tool_manager.get_tool("removed_tool") is None
    assert (
        mcp._tool_manager.get_tool("enabled_tool").description
        == "overridden description"
    )


@pytest.mark.asyncio
async def test_local_search_project_root_updates_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_result = [{"name": "foo", "kind": "def", "file": "Foo.lean"}]
    project_dir = _make_project(tmp_path / "proj")

    called: dict[str, Path] = {}

    def fake_search(*, query: str, limit: int, project_root: Path, path_policy):
        called["query"] = query
        called["limit"] = limit
        called["root"] = project_root
        called["policy_root"] = path_policy.project_root
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
        "policy_root": project_dir.resolve(),
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
async def test_local_search_rejects_remote_project_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(server, "_RG_AVAILABLE", True)
    current_project = _make_project(tmp_path / "proj1")
    other_project = _make_project(tmp_path / "proj2")
    ctx = _make_ctx(
        lean_project_path=current_project,
        active_transport="streamable-http",
        project_switching_allowed=False,
    )

    with pytest.raises(server.LocalSearchError, match="Project switching is disabled"):
        await server.local_search(ctx=ctx, query="foo", project_root=str(other_project))


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


class _MultiAttemptClient:
    """Async-client fake: records that the real document is never mutated."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.update_calls: list[tuple[str, str]] = []
        self.diagnostics_calls = 0

    async def reload_from_disk(self, path: str, wait: bool = False):
        return types.SimpleNamespace(text=self.text)

    async def diagnostics(self, _path: str, fresh: bool = True, timeout=None):
        from leanclient.aio import DiagnosticsReport

        self.diagnostics_calls += 1
        return DiagnosticsReport(items=[], version=1)

    async def update(self, path: str, text: str, wait: bool = False):
        self.update_calls.append((path, text))


class _FakePool:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def run_texts(self, texts, want_goal_at=None, timeout=None):
        from leanclient.aio import DiagnosticsReport, GoalResult
        from leanclient.aio.scratch import TrialResult

        self.texts.extend(texts)
        return [
            TrialResult(
                body=t,
                diagnostics=DiagnosticsReport(items=[], version=1),
                goal=GoalResult(status="complete"),
            )
            for t in texts
        ]


@pytest.mark.asyncio
async def test_multi_attempt_runs_on_scratch_pool_not_user_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Snippets are tried on scratch documents; the user's open document is
    never edited (no update/didChange on it), so there is nothing to restore."""
    project = _make_project(tmp_path / "proj")
    fake_client = _MultiAttemptClient("theorem foo : True := by\n  sorry\n")
    fake_pool = _FakePool()
    ctx = _make_ctx(lean_project_path=project)
    ctx.request_context.lifespan_context.client = fake_client

    monkeypatch.setattr(server, "setup_client_for_file", _async_setup("Foo.lean"))
    monkeypatch.setattr(server, "get_scratch_pool", lambda _ctx: fake_pool)

    result = await server._multi_attempt_lsp(
        ctx, str(project / "Foo.lean"), line=2, snippets=["trivial", "simp"]
    )

    assert [item.snippet for item in result.items] == ["trivial", "simp"]
    assert fake_client.update_calls == []  # the real document was never touched
    assert len(fake_pool.texts) == 2
    assert all("theorem foo" in t for t in fake_pool.texts)
    assert "trivial" in fake_pool.texts[0] and "simp" in fake_pool.texts[1]


@pytest.mark.asyncio
async def test_multi_attempt_diffs_new_diagnostics_against_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A diagnostic introduced by a snippet at a distant line is reported;
    pre-existing baseline diagnostics are not re-reported."""
    from leanclient.aio import DiagnosticsReport, GoalResult
    from leanclient.aio.scratch import TrialResult

    def _diag(line: int, message: str, severity: int = 1) -> dict:
        return {
            "severity": severity,
            "message": message,
            "range": {
                "start": {"line": line, "character": 0},
                "end": {"line": line, "character": 4},
            },
        }

    pre_existing = _diag(20, "pre-existing error")

    class _Client(_MultiAttemptClient):
        async def diagnostics(self, _path, fresh=True, timeout=None):
            return DiagnosticsReport(items=[pre_existing], version=1)

    class _Pool:
        async def run_texts(self, texts, want_goal_at=None, timeout=None):
            return [
                TrialResult(
                    body=texts[0],
                    diagnostics=DiagnosticsReport(
                        items=[pre_existing, _diag(15, "distant new error")],
                        version=2,
                    ),
                    goal=GoalResult(status="goals", goals=["⊢ True"]),
                )
            ]

    project = _make_project(tmp_path / "proj")
    lines = "\n".join(["-- filler"] * 30)
    fake_client = _Client(f"theorem foo : True := by\n  sorry\n{lines}\n")
    ctx = _make_ctx(lean_project_path=project)
    ctx.request_context.lifespan_context.client = fake_client

    monkeypatch.setattr(server, "setup_client_for_file", _async_setup("Foo.lean"))
    monkeypatch.setattr(server, "get_scratch_pool", lambda _ctx: _Pool())

    result = await server._multi_attempt_lsp(
        ctx, str(project / "Foo.lean"), line=2, snippets=["trivial"]
    )

    messages = [d.message for d in result.items[0].diagnostics]
    assert "distant new error" in messages
    assert "pre-existing error" not in messages
    assert result.items[0].goals == ["⊢ True"]


@pytest.mark.asyncio
async def test_declaration_file_sanitizes_dependency_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _make_project(tmp_path / "proj")
    dep_file = _make_dependency(project, tmp_path / "deps" / "mathlib")

    class FakeClient:
        project_path = str(project)

        async def reload_from_disk(self, _path: str, wait: bool = False):
            return None

        def content(self, _path: str) -> str:
            return "dep"

        async def goto(self, kind: str, _path: str, _line: int, _column: int):
            assert kind == "declaration"
            return [
                {
                    "path": str(dep_file),
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 10},
                    },
                }
            ]

    ctx = _make_ctx(lean_project_path=project)
    ctx.request_context.lifespan_context.client = FakeClient()
    monkeypatch.setattr(server, "setup_client_for_file", _async_setup("Main.lean"))

    result = await server.declaration_file(
        ctx=ctx, file_path=str(project / "Main.lean"), symbol="dep"
    )

    assert result.file_path == ".lake/packages/mathlib/Mathlib/Foo.lean"
    assert "theorem dep" in result.content


@pytest.mark.asyncio
async def test_references_sanitize_paths_and_skip_outside_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _make_project(tmp_path / "proj")
    dep_file = _make_dependency(project, tmp_path / "deps" / "mathlib")
    outside_file = tmp_path / "outside" / "Leak.lean"
    outside_file.parent.mkdir(parents=True)
    outside_file.write_text("def leak : Nat := 0\n")

    class FakeClient:
        project_path = str(project)

        async def reload_from_disk(self, _path: str, wait: bool = False):
            return None

        async def references(
            self,
            _path: str,
            _line: int,
            _column: int,
            include_declaration: bool = True,
            max_results=None,
            fresh: bool = True,
        ):
            return [
                {
                    "path": str(dep_file),
                    "range": {
                        "start": {"line": 2, "character": 3},
                        "end": {"line": 2, "character": 9},
                    },
                },
                {
                    "path": str(outside_file),
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 4},
                    },
                },
            ]

    ctx = _make_ctx(lean_project_path=project)
    ctx.request_context.lifespan_context.client = FakeClient()
    monkeypatch.setattr(server, "setup_client_for_file", _async_setup("Main.lean"))

    result = await server.references(
        ctx=ctx,
        file_path=str(project / "Main.lean"),
        line=1,
        column=1,
    )

    assert len(result.items) == 1
    assert result.items[0].file_path == ".lake/packages/mathlib/Mathlib/Foo.lean"
    assert result.items[0].line == 3
    assert result.items[0].column == 4


@pytest.mark.asyncio
async def test_verify_theorem_rejects_invalid_theorem_name() -> None:
    ctx = _make_ctx()

    with pytest.raises(server.LeanToolError, match="Invalid theorem name"):
        await server.verify_theorem(
            ctx=ctx, file_path="Foo.lean", theorem_name="bad name"
        )


@pytest.mark.asyncio
async def test_profile_proof_rejects_paths_outside_policy(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "proj")
    outside_file = tmp_path / "outside" / "Leak.lean"
    outside_file.parent.mkdir(parents=True)
    outside_file.write_text("theorem leak : True := by trivial\n")
    ctx = _make_ctx(lean_project_path=project)

    with pytest.raises(server.LeanToolError, match="outside the active Lean project"):
        await server.profile_proof(ctx=ctx, file_path=str(outside_file), line=1)


# ---------------------------------------------------------------------------
# Severity filtering in _process_diagnostics / diagnostic_messages
# ---------------------------------------------------------------------------

_MIXED_DIAGNOSTICS = [
    {
        "severity": 1,
        "message": "unknown identifier",
        "range": {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 5},
        },
    },
    {
        "severity": 2,
        "message": "unused variable",
        "range": {
            "start": {"line": 1, "character": 0},
            "end": {"line": 1, "character": 3},
        },
    },
    {
        "severity": 3,
        "message": "declaration uses sorry",
        "range": {
            "start": {"line": 2, "character": 0},
            "end": {"line": 2, "character": 4},
        },
    },
    {
        "severity": 4,
        "message": "consider using simp",
        "range": {
            "start": {"line": 3, "character": 0},
            "end": {"line": 3, "character": 3},
        },
    },
]


def test_process_diagnostics_no_severity_filter_returns_all() -> None:
    result = server._process_diagnostics(
        _MIXED_DIAGNOSTICS, build_success=True, severity=None
    )
    assert [d.severity for d in result.items] == ["error", "warning", "info", "hint"]


def test_process_diagnostics_filter_errors_only() -> None:
    result = server._process_diagnostics(
        _MIXED_DIAGNOSTICS, build_success=True, severity=DiagnosticSeverity.error
    )
    assert len(result.items) == 1
    assert result.items[0].severity == "error"
    assert result.items[0].message == "unknown identifier"


def test_process_diagnostics_filter_warnings_only() -> None:
    result = server._process_diagnostics(
        _MIXED_DIAGNOSTICS, build_success=True, severity=DiagnosticSeverity.warning
    )
    assert len(result.items) == 1
    assert result.items[0].severity == "warning"
    assert result.items[0].message == "unused variable"


def test_process_diagnostics_filter_info_only() -> None:
    result = server._process_diagnostics(
        _MIXED_DIAGNOSTICS, build_success=True, severity=DiagnosticSeverity.info
    )
    assert len(result.items) == 1
    assert result.items[0].severity == "info"


def test_process_diagnostics_filter_hint_only() -> None:
    result = server._process_diagnostics(
        _MIXED_DIAGNOSTICS, build_success=True, severity=DiagnosticSeverity.hint
    )
    assert len(result.items) == 1
    assert result.items[0].severity == "hint"


def test_process_diagnostics_filter_no_matches_returns_empty() -> None:
    error_only = [_MIXED_DIAGNOSTICS[0]]
    result = server._process_diagnostics(
        error_only, build_success=True, severity=DiagnosticSeverity.warning
    )
    assert result.items == []


def test_process_diagnostics_build_failure_excluded_regardless_of_filter() -> None:
    """Build stderr blobs at (1,1) should be excluded even when severity filter matches."""
    # "lake setup-file" triggers is_build_stderr
    build_diag = {
        "severity": 1,
        "message": "lake setup-file some/dep/path",
        "range": {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 0},
        },
    }
    result = server._process_diagnostics(
        [build_diag], build_success=False, severity=DiagnosticSeverity.error
    )
    assert result.items == []
    assert result.success is False


def test_process_diagnostics_accepts_plain_string_severity() -> None:
    """The tool now passes severity as a plain string (Literal), not an enum.

    See issue #185: the parameter type changed from a str-Enum to a Literal so
    the emitted JSON schema is Gemini/Vertex compatible.
    """
    result = server._process_diagnostics(
        _MIXED_DIAGNOSTICS, build_success=True, severity="warning"
    )
    assert len(result.items) == 1
    assert result.items[0].severity == "warning"


def test_diagnostic_messages_severity_schema_is_vertex_compatible() -> None:
    """Regression test for #185.

    Google Gemini/Vertex function-calling rejects a parameter schema that lacks
    an explicit top-level ``type`` field, and cannot resolve JSON-Schema
    ``$ref``/``$defs``. The ``severity`` parameter of ``lean_diagnostic_messages``
    must therefore expose a top-level ``type`` and contain no ``$ref`` while
    still constraining the value to the four severity levels.
    """
    tools = asyncio.run(server.mcp.list_tools())
    tool = next(t for t in tools if t.name == "lean_diagnostic_messages")
    severity = tool.inputSchema["properties"]["severity"]
    severity_json = json.dumps(severity)

    assert severity.get("type") == "string"
    assert "$ref" not in severity_json
    assert "DiagnosticSeverity" not in json.dumps(tool.inputSchema.get("$defs", {}))
    for level in ("error", "warning", "info", "hint"):
        assert level in severity_json


class _DiagClient:
    def __init__(self, items=None):
        from leanclient.aio import DiagnosticsReport

        self._report = DiagnosticsReport(items=items or [], version=1)

    async def reload_from_disk(self, _path: str, wait: bool = False):
        return None

    async def diagnostics(self, _path: str, fresh: bool = True, timeout=None):
        return self._report


@pytest.mark.asyncio
async def test_diagnostic_messages_passes_severity_to_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """diagnostic_messages tool forwards the severity parameter to _process_diagnostics."""
    captured: dict = {}

    def fake_process(diagnostics, build_success, severity=None, timed_out=False):
        captured["severity"] = severity
        from lean_lsp_mcp.models import DiagnosticsResult

        return DiagnosticsResult(success=build_success, items=[])

    monkeypatch.setattr(server, "_process_diagnostics", fake_process)
    monkeypatch.setattr(server, "setup_client_for_file", _async_setup("Foo.lean"))

    ctx = _make_ctx()
    ctx.request_context.lifespan_context.client = _DiagClient(
        items=[
            {
                "severity": 2,
                "message": "unused",
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 3},
                },
            }
        ]
    )

    await server.diagnostic_messages(
        ctx=ctx, file_path="/abs/Foo.lean", severity=DiagnosticSeverity.warning
    )

    assert captured["severity"] == DiagnosticSeverity.warning


@pytest.mark.asyncio
async def test_diagnostic_messages_default_severity_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_process(diagnostics, build_success, severity=None, timed_out=False):
        captured["severity"] = severity
        from lean_lsp_mcp.models import DiagnosticsResult

        return DiagnosticsResult(success=build_success, items=[])

    monkeypatch.setattr(server, "_process_diagnostics", fake_process)
    monkeypatch.setattr(server, "setup_client_for_file", _async_setup("Foo.lean"))

    ctx = _make_ctx()
    ctx.request_context.lifespan_context.client = _DiagClient()

    await server.diagnostic_messages(ctx=ctx, file_path="/abs/Foo.lean")

    assert captured["severity"] is None


class _GoalClient:
    """Async fake for the goal tool, parameterized by GoalResult."""

    def __init__(self, text: str, result) -> None:
        from leanclient.aio import GoalResult

        self._text = text
        self._result = result if result is not None else GoalResult(status="no_goal")
        self.goal_calls = 0

    async def reload_from_disk(self, _path: str, wait: bool = False):
        return None

    def content(self, _path: str) -> str:
        return self._text

    async def goal(self, _path: str, _line: int, _column: int, fresh: bool = True):
        self.goal_calls += 1
        return self._result


_SAMPLE_TEXT = "import Mathlib\n\ntheorem sample_goal : True := by\n  trivial\n"


@pytest.mark.asyncio
async def test_goal_reports_open_goals(monkeypatch: pytest.MonkeyPatch) -> None:
    from leanclient.aio import GoalResult

    monkeypatch.setattr(server, "setup_client_for_file", _async_setup("GoalSample.lean"))
    ctx = _make_ctx()
    fake = _GoalClient(_SAMPLE_TEXT, GoalResult(status="goals", goals=["⊢ True"]))
    ctx.request_context.lifespan_context.client = fake

    result = await server.goal(ctx, file_path="/abs/GoalSample.lean", line=4, column=3)

    assert result.goals == ["⊢ True"]
    assert result.status == "goals"
    assert fake.goal_calls == 1


@pytest.mark.asyncio
async def test_goal_distinguishes_complete_from_no_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flagship ambiguity fix: an empty goal list means 'complete' only
    when elaboration reached the position; a position with no proof state
    reports 'no_goal_at_position'. Both have goals == []."""
    from leanclient.aio import GoalResult

    monkeypatch.setattr(server, "setup_client_for_file", _async_setup("GoalSample.lean"))

    ctx = _make_ctx()
    ctx.request_context.lifespan_context.client = _GoalClient(
        _SAMPLE_TEXT, GoalResult(status="complete")
    )
    done = await server.goal(ctx, file_path="/abs/GoalSample.lean", line=4, column=3)
    assert done.status == "complete"
    assert done.goals == []

    ctx2 = _make_ctx()
    ctx2.request_context.lifespan_context.client = _GoalClient(
        _SAMPLE_TEXT, GoalResult(status="no_goal")
    )
    nowhere = await server.goal(ctx2, file_path="/abs/GoalSample.lean", line=1, column=1)
    assert nowhere.status == "no_goal_at_position"
    assert nowhere.goals == []


@pytest.mark.asyncio
async def test_goal_structured_format_accepts_structured_goals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from leanclient.aio import GoalResult

    monkeypatch.setattr(server, "setup_client_for_file", _async_setup("GoalSample.lean"))
    ctx = _make_ctx()
    ctx.request_context.lifespan_context.client = _GoalClient(
        "import Mathlib\n\ntheorem sample_goal (x : Nat) (h : x = 0) : x = 0 := by\n  exact h\n",
        GoalResult(status="goals", goals=["x : Nat\nh : x = 0\n⊢ x = 0"]),
    )

    result = await server.goal(
        ctx,
        file_path="/abs/GoalSample.lean",
        line=4,
        column=3,
        format="structured",
    )

    assert result.goals is not None
    structured_goal = result.goals[0]
    assert not isinstance(structured_goal, str)
    assert structured_goal.model_dump() == {
        "context": [
            {"name": "x", "type": "Nat"},
            {"name": "h", "type": "x = 0"},
        ],
        "goal": "x = 0",
        "status": "open",
        "pretty": "x : Nat\nh : x = 0\n⊢ x = 0",
    }


@pytest.mark.asyncio
async def test_multi_attempt_repl_does_not_autodiscover_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("unexpected")

    ctx = _make_ctx()
    lifespan = ctx.request_context.lifespan_context
    lifespan.repl_enabled = True
    lifespan.repl = None
    monkeypatch.setattr(server, "resolve_file_path", fail_resolve)

    result = await server._multi_attempt_repl(
        ctx,
        file_path="/abs/Foo.lean",
        line=1,
        snippets=["trivial"],
    )

    assert result is None

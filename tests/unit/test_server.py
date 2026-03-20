from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError
import importlib
import json
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lean_lsp_mcp import server
from lean_lsp_mcp.models import DiagnosticSeverity


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


@pytest.fixture(autouse=True)
def _clear_optional_runtime_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEAN_LSP_MCP_ACTIVE_TRANSPORT", raising=False)
    monkeypatch.delenv("LEAN_PROJECT_PATH", raising=False)
    monkeypatch.delenv("LEAN_LSP_MCP_TOKEN", raising=False)
    monkeypatch.delenv("LEAN_LOOGLE_LOCAL", raising=False)
    monkeypatch.delenv("LEAN_REPL", raising=False)


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
async def test_app_lifespan_requires_project_path_for_streamable_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEAN_LSP_MCP_ACTIVE_TRANSPORT", "streamable-http")

    with pytest.raises(ValueError, match="`LEAN_PROJECT_PATH` is required"):
        async with server.app_lifespan(object()):
            pass


@pytest.mark.asyncio
async def test_app_lifespan_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEAN_LOG_LEVEL", raising=False)

    dummy_client = DummyClient()

    async with server.app_lifespan(object()) as context:
        context.client = dummy_client

    assert dummy_client.closed_calls == 1


@pytest.mark.asyncio
async def test_app_lifespan_suppresses_client_close_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEAN_LOG_LEVEL", raising=False)

    dummy_client = _FailingCloseClient()

    async with server.app_lifespan(object()) as context:
        context.client = dummy_client

    assert dummy_client.close_calls == 1


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeClient(_BaseMultiAttemptClient):
        def get_file_content(self, _path: str) -> str:
            return "buffer-content"

    project = _make_project(tmp_path / "proj")
    target = project / "Foo.lean"
    target.write_text("theorem foo : True := by trivial\n")
    fake_client = FakeClient()
    ctx = _make_ctx(lean_project_path=project)
    ctx.request_context.lifespan_context.client = fake_client

    monkeypatch.setattr(server, "setup_client_for_file", lambda _ctx, _path: "Foo.lean")
    monkeypatch.setattr(server, "get_file_contents", lambda _path: "original")

    result = server._multi_attempt_lsp(ctx, str(target), line=1, snippets=[])

    assert result.items == []
    assert fake_client.restore_calls == [("Foo.lean", "buffer-content")]
    assert fake_client.open_calls == [("Foo.lean", False), ("Foo.lean", True)]


def test_multi_attempt_restore_falls_back_to_disk_on_buffer_read_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeClient(_BaseMultiAttemptClient):
        def get_file_content(self, _path: str) -> str:
            raise RuntimeError("buffer unavailable")

    project = _make_project(tmp_path / "proj")
    target = project / "Foo.lean"
    target.write_text("theorem foo : True := by trivial\n")
    fake_client = FakeClient()
    ctx = _make_ctx(lean_project_path=project)
    ctx.request_context.lifespan_context.client = fake_client

    monkeypatch.setattr(server, "setup_client_for_file", lambda _ctx, _path: "Foo.lean")
    monkeypatch.setattr(server, "get_file_contents", lambda _path: "disk-content")

    result = server._multi_attempt_lsp(ctx, str(target), line=1, snippets=[])

    assert result.items == []
    assert fake_client.restore_calls == [("Foo.lean", "disk-content")]


def test_declaration_file_sanitizes_dependency_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _make_project(tmp_path / "proj")
    dep_file = _make_dependency(project, tmp_path / "deps" / "mathlib")

    class FakeClient:
        def open_file(self, _path: str) -> None:
            return

        def get_file_content(self, _path: str) -> str:
            return "dep"

        def get_declarations(self, _path: str, _line: int, _column: int) -> list[dict]:
            return [{"uri": "dep-uri"}]

        def _uri_to_abs(self, uri: str) -> str:
            assert uri == "dep-uri"
            return str(dep_file)

    ctx = _make_ctx(lean_project_path=project)
    ctx.request_context.lifespan_context.client = FakeClient()
    monkeypatch.setattr(
        server, "setup_client_for_file", lambda _ctx, _path: "Main.lean"
    )

    result = server.declaration_file(
        ctx=ctx, file_path=str(project / "Main.lean"), symbol="dep"
    )

    assert result.file_path == ".lake/packages/mathlib/Mathlib/Foo.lean"
    assert "theorem dep" in result.content


def test_references_sanitize_paths_and_skip_outside_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _make_project(tmp_path / "proj")
    dep_file = _make_dependency(project, tmp_path / "deps" / "mathlib")
    outside_file = tmp_path / "outside" / "Leak.lean"
    outside_file.parent.mkdir(parents=True)
    outside_file.write_text("def leak : Nat := 0\n")

    class FakeClient:
        def open_file(self, _path: str) -> None:
            return

        def get_diagnostics(self, _path: str) -> list[dict]:
            return []

        def get_references(
            self, _path: str, _line: int, _column: int, include_declaration: bool = True
        ) -> list[dict]:
            _ = include_declaration
            return [
                {
                    "uri": "dep-uri",
                    "range": {
                        "start": {"line": 2, "character": 3},
                        "end": {"line": 2, "character": 9},
                    },
                },
                {
                    "uri": "outside-uri",
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 4},
                    },
                },
            ]

        def _uri_to_abs(self, uri: str) -> str:
            if uri == "dep-uri":
                return str(dep_file)
            if uri == "outside-uri":
                return str(outside_file)
            raise AssertionError(f"unexpected uri: {uri}")

    ctx = _make_ctx(lean_project_path=project)
    ctx.request_context.lifespan_context.client = FakeClient()
    monkeypatch.setattr(
        server, "setup_client_for_file", lambda _ctx, _path: "Main.lean"
    )

    result = server.references(
        ctx=ctx,
        file_path=str(project / "Main.lean"),
        line=1,
        column=1,
    )

    assert len(result.items) == 1
    assert result.items[0].file_path == ".lake/packages/mathlib/Mathlib/Foo.lean"
    assert result.items[0].line == 3
    assert result.items[0].column == 4


def test_verify_theorem_rejects_invalid_theorem_name() -> None:
    ctx = _make_ctx()

    with pytest.raises(server.LeanToolError, match="Invalid theorem name"):
        server.verify_theorem(ctx=ctx, file_path="Foo.lean", theorem_name="bad name")


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


def test_diagnostic_messages_passes_severity_to_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """diagnostic_messages tool forwards the severity parameter to _process_diagnostics."""
    captured: dict = {}

    def fake_process(diagnostics, build_success, severity=None):
        captured["severity"] = severity
        from lean_lsp_mcp.models import DiagnosticsResult

        return DiagnosticsResult(success=build_success, items=[])

    class FakeDiagResult:
        diagnostics = [
            {
                "severity": 2,
                "message": "unused",
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 3},
                },
            }
        ]
        success = True

    class FakeClient:
        def open_file(self, *_a, **_kw):
            pass

        def get_diagnostics(self, *_a, **_kw):
            return FakeDiagResult()

    monkeypatch.setattr(server, "_process_diagnostics", fake_process)
    monkeypatch.setattr(server, "setup_client_for_file", lambda _ctx, _path: "Foo.lean")

    ctx = _make_ctx()
    ctx.request_context.lifespan_context.client = FakeClient()

    server.diagnostic_messages(
        ctx=ctx, file_path="/abs/Foo.lean", severity=DiagnosticSeverity.warning
    )

    assert captured["severity"] == DiagnosticSeverity.warning


def test_diagnostic_messages_default_severity_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_process(diagnostics, build_success, severity=None):
        captured["severity"] = severity
        from lean_lsp_mcp.models import DiagnosticsResult

        return DiagnosticsResult(success=build_success, items=[])

    class FakeDiagResult:
        diagnostics = []
        success = True

    class FakeClient:
        def open_file(self, *_a, **_kw):
            pass

        def get_diagnostics(self, *_a, **_kw):
            return FakeDiagResult()

    monkeypatch.setattr(server, "_process_diagnostics", fake_process)
    monkeypatch.setattr(server, "setup_client_for_file", lambda _ctx, _path: "Foo.lean")

    ctx = _make_ctx()
    ctx.request_context.lifespan_context.client = FakeClient()

    server.diagnostic_messages(ctx=ctx, file_path="/abs/Foo.lean")

    assert captured["severity"] is None


def test_goal_retries_after_cold_file_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.goal_calls = 0
            self.diagnostic_calls: list[tuple[str, float]] = []
            self.open_calls: list[tuple[str, bool]] = []

        def open_file(self, path: str, force_reopen: bool = False, **_kw) -> None:
            self.open_calls.append((path, force_reopen))
            return

        def get_file_content(self, _path: str) -> str:
            return "import Mathlib\n\ntheorem sample_goal : True := by\n  trivial\n"

        def get_goal(self, _path: str, _line: int, _column: int) -> dict:
            self.goal_calls += 1
            if self.goal_calls == 1:
                raise FuturesTimeoutError()
            return {"goals": ["⊢ True"]}

        def get_diagnostics(
            self,
            path: str,
            *,
            inactivity_timeout: float,
        ):
            self.diagnostic_calls.append((path, inactivity_timeout))
            return types.SimpleNamespace(diagnostics=[], success=True)

    monkeypatch.setattr(
        server, "setup_client_for_file", lambda _ctx, _path: "GoalSample.lean"
    )

    ctx = _make_ctx()
    fake_client = FakeClient()
    ctx.request_context.lifespan_context.client = fake_client

    result = server.goal(ctx, file_path="/abs/GoalSample.lean", line=4, column=3)

    assert result.goals == ["⊢ True"]
    assert fake_client.goal_calls == 2
    assert fake_client.open_calls == [("GoalSample.lean", False)]
    assert fake_client.diagnostic_calls == [("GoalSample.lean", 30.0)]


def test_goal_returns_no_goals_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.goal_calls = 0
            self.diagnostic_calls = 0
            self.open_calls: list[tuple[str, bool]] = []

        def open_file(self, path: str, force_reopen: bool = False, **_kw) -> None:
            self.open_calls.append((path, force_reopen))

        def get_file_content(self, _path: str) -> str:
            return "import Mathlib\n\ntheorem sample_goal : True := by\n  trivial\n"

        def get_goal(self, _path: str, _line: int, _column: int) -> None:
            self.goal_calls += 1
            return None

        def get_diagnostics(self, *_a, **_kw):
            self.diagnostic_calls += 1
            return types.SimpleNamespace(diagnostics=[], success=True)

    monkeypatch.setattr(
        server, "setup_client_for_file", lambda _ctx, _path: "GoalSample.lean"
    )

    ctx = _make_ctx()
    fake_client = FakeClient()
    ctx.request_context.lifespan_context.client = fake_client

    result = server.goal(ctx, file_path="/abs/GoalSample.lean", line=4, column=3)

    assert result.goals == []
    assert fake_client.goal_calls == 1
    assert fake_client.open_calls == [("GoalSample.lean", False)]
    assert fake_client.diagnostic_calls == 0


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

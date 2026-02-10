from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from lean_lsp_mcp import server


class DummyClient:
    def __init__(self) -> None:
        self.closed_calls = 0

    def close(self) -> None:
        self.closed_calls += 1


def _make_ctx(
    rate_limit: dict[str, list[int]] | None = None,
    *,
    lean_project_path: Path | None = None,
    strict_project_root: bool = False,
) -> types.SimpleNamespace:
    context = server.AppContext(
        lean_project_path=lean_project_path,
        client=None,
        rate_limit=rate_limit or {"test": []},
        lean_search_available=True,
        strict_project_root=strict_project_root,
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


def test_load_tool_description_overrides_inline_and_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    overrides_file = tmp_path / "tool_descriptions.json"
    overrides_file.write_text(
        json.dumps(
            {
                "lean_build": "Build tool from file",
                "lean_goal": "Goal tool from file",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "LEAN_MCP_TOOL_DESCRIPTIONS",
        json.dumps({"lean_build": "Build tool from env"}),
    )
    monkeypatch.setenv("LEAN_MCP_TOOL_DESCRIPTIONS_FILE", str(overrides_file))

    overrides = server._load_tool_description_overrides()
    assert overrides["lean_build"] == "Build tool from file"
    assert overrides["lean_goal"] == "Goal tool from file"


def test_apply_tool_configuration_disables_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp = server.FastMCP(name="test")

    @mcp.tool("enabled_tool")
    def enabled_tool() -> str:
        """enabled description"""
        return "ok"

    @mcp.tool("removed_tool")
    def removed_tool() -> str:
        """removed description"""
        return "ok"

    monkeypatch.setenv("LEAN_MCP_DISABLED_TOOLS", "removed_tool")
    monkeypatch.setenv(
        "LEAN_MCP_TOOL_DESCRIPTIONS",
        json.dumps({"enabled_tool": "overridden description"}),
    )

    server.apply_tool_configuration(mcp)

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
async def test_local_search_blocks_root_escape_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(server, "_RG_AVAILABLE", True)

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    ctx = _make_ctx(lean_project_path=project_dir, strict_project_root=True)

    with pytest.raises(server.LocalSearchError) as exc_info:
        await server.local_search(
            ctx=ctx,
            query="foo",
            project_root=str(outside_dir),
        )

    assert "outside configured LEAN_PROJECT_PATH" in str(exc_info.value)

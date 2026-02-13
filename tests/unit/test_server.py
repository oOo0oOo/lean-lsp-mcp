from __future__ import annotations

import types
import urllib.error
from pathlib import Path

import pytest

from lean_lsp_mcp import server


class DummyClient:
    def __init__(self) -> None:
        self.closed_calls = 0

    def close(self) -> None:
        self.closed_calls += 1


def _make_ctx(rate_limit: dict[str, list[int]] | None = None) -> types.SimpleNamespace:
    default_rate_limit = {
        "test": [],
        "leansearch": [],
        "loogle": [],
        "leanfinder": [],
        "lean_state_search": [],
        "hammer_premise": [],
    }
    context = server.AppContext(
        lean_project_path=None,
        client=None,
        rate_limit=rate_limit or default_rate_limit,
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


def test_local_search_module_name_for_package_path() -> None:
    module_name = server._local_search_module_name(
        ".lake/packages/mathlib/Mathlib/Algebra/Group/Defs.lean"
    )
    assert module_name == "Mathlib.Algebra.Group.Defs"


@pytest.mark.asyncio
async def test_leansearch_local_fallback_on_remote_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    ctx = _make_ctx()
    ctx.request_context.lifespan_context.lean_project_path = project_dir

    async def failing_urlopen(req, timeout: float = 10.0):
        _ = req, timeout
        raise urllib.error.URLError("offline")

    def fake_local_search(*, query: str, limit: int, project_root: Path):
        assert query == "Nat.succ"
        assert limit == 2
        assert project_root == project_dir.resolve()
        return [
            {"name": "Nat.succ", "kind": "def", "file": "Mathlib/Init/Prelude.lean"}
        ]

    monkeypatch.setattr(server, "_urlopen_json", failing_urlopen)
    monkeypatch.setattr(server, "_RG_AVAILABLE", True)
    monkeypatch.setattr(server, "lean_local_search", fake_local_search)

    result = await server.leansearch(ctx=ctx, query="Nat.succ", num_results=1)
    assert result.items == [
        server.LeanSearchResult(
            name="Nat.succ",
            module_name="Mathlib.Init.Prelude",
            kind="def",
            type=None,
        )
    ]


@pytest.mark.asyncio
async def test_leansearch_disable_local_fallback_raises_remote_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_urlopen(req, timeout: float = 10.0):
        _ = req, timeout
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(server, "_urlopen_json", failing_urlopen)

    with pytest.raises(server.LeanToolError, match="leansearch.net request failed"):
        await server.leansearch(
            ctx=_make_ctx(),
            query="Nat.succ",
            num_results=1,
            local_fallback=False,
        )


@pytest.mark.asyncio
async def test_leansearch_project_root_override_for_local_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    async def failing_urlopen(req, timeout: float = 10.0):
        _ = req, timeout
        raise urllib.error.URLError("offline")

    captured: dict[str, object] = {}

    def fake_local_search(*, query: str, limit: int, project_root: Path):
        captured["query"] = query
        captured["limit"] = limit
        captured["project_root"] = project_root
        return [{"name": "demo", "kind": "lemma", "file": "Demo.lean"}]

    ctx = _make_ctx()
    monkeypatch.setattr(server, "_urlopen_json", failing_urlopen)
    monkeypatch.setattr(server, "_RG_AVAILABLE", True)
    monkeypatch.setattr(server, "lean_local_search", fake_local_search)

    result = await server.leansearch(
        ctx=ctx,
        query="demo",
        num_results=1,
        project_root=str(project_dir),
    )

    assert len(result.items) == 1
    assert captured["query"] == "demo"
    assert captured["project_root"] == project_dir.resolve()
    assert (
        ctx.request_context.lifespan_context.lean_project_path == project_dir.resolve()
    )


def test_leansearch_parse_remote_results_accepts_dict_results_shape() -> None:
    payload = {
        "results": [
            {
                "name": "Nat.add",
                "module_name": "Mathlib.Data.Nat.Basic",
                "kind": "theorem",
                "type": "Nat -> Nat -> Nat",
            },
            {
                "result": {
                    "name": ["Nat", "succ"],
                    "module_name": ["Mathlib", "Init", "Prelude"],
                    "kind": "def",
                    "signature": "Nat -> Nat",
                }
            },
        ]
    }

    items = server._leansearch_parse_remote_results(payload, num_results=5)

    assert items == [
        server.LeanSearchResult(
            name="Nat.add",
            module_name="Mathlib.Data.Nat.Basic",
            kind="theorem",
            type="Nat -> Nat -> Nat",
        ),
        server.LeanSearchResult(
            name="Nat.succ",
            module_name="Mathlib.Init.Prelude",
            kind="def",
            type="Nat -> Nat",
        ),
    ]


def test_leansearch_parse_remote_results_accepts_flat_list_shape() -> None:
    payload = [
        {
            "name": "Nat.zero",
            "module": "Mathlib.Init.Prelude",
            "kind": "def",
            "type": "Nat",
        }
    ]

    items = server._leansearch_parse_remote_results(payload, num_results=1)

    assert items == [
        server.LeanSearchResult(
            name="Nat.zero",
            module_name="Mathlib.Init.Prelude",
            kind="def",
            type="Nat",
        )
    ]

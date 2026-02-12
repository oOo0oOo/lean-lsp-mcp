from __future__ import annotations

import types
import urllib.error

import pytest

from lean_lsp_mcp import server


def _make_ctx(
    *,
    leanexplore_local_enabled: bool = False,
    leanexplore_service: object | None = None,
) -> types.SimpleNamespace:
    lifespan_context = types.SimpleNamespace(
        rate_limit={"leanexplore": []},
        leanexplore_local_enabled=leanexplore_local_enabled,
        leanexplore_service=leanexplore_service,
    )
    request_context = types.SimpleNamespace(lifespan_context=lifespan_context)
    return types.SimpleNamespace(request_context=request_context)


def _sample_item() -> dict:
    return {
        "id": 42,
        "primary_declaration": {"lean_name": "Nat.add"},
        "source_file": "Mathlib/Data/Nat/Basic.lean",
        "range_start_line": 10,
        "statement_text": "theorem Nat.add : Nat -> Nat -> Nat",
        "docstring": "Addition on natural numbers.",
        "informal_description": "Natural number addition.",
    }


@pytest.mark.asyncio
async def test_leanexplore_search_builds_url_and_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_urlopen_json(req, timeout: float = 10.0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        return {
            "results": [_sample_item()],
            "count": 1,
            "total_candidates_considered": 1,
            "processing_time_ms": 5,
        }

    monkeypatch.setenv("LEAN_EXPLORE_BASE_URL", "https://example.test/api/v1")
    monkeypatch.setenv("LEAN_EXPLORE_API_KEY", "token123")
    monkeypatch.setattr(server, "_urlopen_json", fake_urlopen_json)

    ctx = _make_ctx()
    result = await server.leanexplore_search(
        ctx=ctx,
        query="Nat",
        package_filters=["Mathlib", "Batteries"],
        limit=1,
    )

    assert len(result.items) == 1
    assert result.items[0].lean_name == "Nat.add"

    url = captured["url"]
    assert "q=Nat" in url
    assert "limit=1" in url
    assert "pkg=Mathlib" in url
    assert "pkg=Batteries" in url
    assert "packages=Mathlib%2CBatteries" in url

    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers.get("authorization") == "Bearer token123"


@pytest.mark.asyncio
async def test_leanexplore_default_rerank_top_can_be_overridden_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LocalService:
        async def search(self, *, query: str, limit: int, rerank_top: int | None):
            assert query == "Nat"
            assert limit == 1
            assert rerank_top == 7
            return {"results": [_sample_item()]}

    monkeypatch.setenv("LEAN_EXPLORE_RERANK_TOP", "7")
    ctx = _make_ctx(
        leanexplore_local_enabled=True,
        leanexplore_service=LocalService(),
    )

    result = await server.leanexplore_search(ctx=ctx, query="Nat", limit=1)
    assert result.items[0].id == 42


def test_leanexplore_default_rerank_top_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEAN_EXPLORE_RERANK_TOP", "-1")
    with pytest.raises(server.LeanToolError, match="LEAN_EXPLORE_RERANK_TOP"):
        server._leanexplore_default_rerank_top()

    monkeypatch.setenv("LEAN_EXPLORE_RERANK_TOP", "abc")
    with pytest.raises(server.LeanToolError, match="LEAN_EXPLORE_RERANK_TOP"):
        server._leanexplore_default_rerank_top()

    monkeypatch.setenv("LEAN_EXPLORE_RERANK_TOP", "none")
    assert server._leanexplore_default_rerank_top() is None


def test_leanexplore_local_service_error_mentions_local_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("lean_explore.search.service", "lean_explore.local.service"):
            raise ModuleNotFoundError("No module named 'lean_explore'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    app_ctx = server.AppContext(
        lean_project_path=None,
        client=None,
        rate_limit={"leanexplore": []},
        lean_search_available=True,
        loogle_manager=None,
        loogle_local_available=False,
        repl=None,
        repl_enabled=False,
        leanexplore_local_enabled=True,
        leanexplore_service=None,
    )

    with pytest.raises(server.LeanToolError) as exc_info:
        server._leanexplore_get_local_service(app_ctx)

    message = str(exc_info.value)
    assert "lean-explore[local]" in message
    assert "lean-explore data fetch" in message


@pytest.mark.asyncio
async def test_leanexplore_get_by_id_prefers_declarations_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_urlopen_json(req, timeout: float = 10.0):
        captured["url"] = req.full_url
        return _sample_item()

    monkeypatch.setenv("LEAN_EXPLORE_BASE_URL", "https://example.test/api/v1")
    monkeypatch.setattr(server, "_urlopen_json", fake_urlopen_json)

    ctx = _make_ctx()
    result = await server.leanexplore_get_by_id(ctx=ctx, group_id=42)

    assert result.id == 42
    assert captured["url"].endswith("/declarations/42")


@pytest.mark.asyncio
async def test_leanexplore_get_by_id_falls_back_to_statement_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    async def fake_urlopen_json(req, timeout: float = 10.0):
        url = req.full_url
        requested_urls.append(url)
        if url.endswith("/declarations/42"):
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return _sample_item()

    monkeypatch.setenv("LEAN_EXPLORE_BASE_URL", "https://example.test/api/v1")
    monkeypatch.setattr(server, "_urlopen_json", fake_urlopen_json)

    ctx = _make_ctx()
    result = await server.leanexplore_get_by_id(ctx=ctx, group_id=42)

    assert result.id == 42
    assert requested_urls[0].endswith("/declarations/42")
    assert requested_urls[1].endswith("/statement_groups/42")


@pytest.mark.asyncio
async def test_leanexplore_dependencies_falls_back_to_declaration_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_urlopen_json(req, timeout: float = 10.0):
        url = req.full_url
        if url.endswith("/declarations/42/dependencies"):
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        if url.endswith("/statement_groups/42/dependencies"):
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        if url.endswith("/declarations/42"):
            item = _sample_item()
            item["dependencies"] = '["Nat.succ", "Nat.zero"]'
            return item
        raise AssertionError(f"Unexpected URL requested: {url}")

    monkeypatch.setenv("LEAN_EXPLORE_BASE_URL", "https://example.test/api/v1")
    monkeypatch.setattr(server, "_urlopen_json", fake_urlopen_json)

    ctx = _make_ctx()
    result = await server.leanexplore_dependencies(ctx=ctx, group_id=42, limit=10)

    assert [item.lean_name for item in result.items] == ["Nat.succ", "Nat.zero"]
    assert all(item.id < 0 for item in result.items)


@pytest.mark.asyncio
async def test_leanexplore_local_v1_style_service_works() -> None:
    class LocalV1Service:
        async def search(
            self,
            *,
            query: str,
            limit: int,
            packages: list[str] | None = None,
            rerank_top: int | None = 50,
        ):
            assert query == "Nat"
            assert limit == 1
            assert packages == ["Mathlib"]
            assert rerank_top == 9
            return {
                "results": [
                    {
                        "id": 123,
                        "name": "Nat.add",
                        "module": "Mathlib.Data.Nat.Basic",
                        "source_text": "theorem Nat.add : Nat -> Nat -> Nat := by ...",
                        "informalization": "Natural number addition.",
                    }
                ]
            }

        async def get_by_id(self, *, declaration_id: int):
            assert declaration_id == 123
            return {
                "id": 123,
                "name": "Nat.add",
                "module": "Mathlib.Data.Nat.Basic",
                "source_text": "theorem Nat.add : Nat -> Nat -> Nat := by ...",
                "dependencies": '["Nat.succ"]',
            }

    ctx = _make_ctx(
        leanexplore_local_enabled=True,
        leanexplore_service=LocalV1Service(),
    )
    search_result = await server.leanexplore_search(
        ctx=ctx,
        query="Nat",
        package_filters=["Mathlib"],
        rerank_top=9,
        limit=1,
    )
    dep_result = await server.leanexplore_dependencies(ctx=ctx, group_id=123, limit=10)

    assert search_result.items[0].id == 123
    assert search_result.items[0].lean_name == "Nat.add"
    assert dep_result.items[0].lean_name == "Nat.succ"


@pytest.mark.asyncio
async def test_leanexplore_search_accepts_legacy_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_urlopen_json(req, timeout: float = 10.0):
        captured["headers"] = dict(req.header_items())
        _ = timeout
        return {"results": [_sample_item()], "count": 1}

    monkeypatch.setenv("LEAN_EXPLORE_BASE_URL", "https://example.test/api/v1")
    monkeypatch.delenv("LEAN_EXPLORE_API_KEY", raising=False)
    monkeypatch.setenv("LEANEXPLORE_API_KEY", "legacy-token")
    monkeypatch.setattr(server, "_urlopen_json", fake_urlopen_json)

    _ = await server.leanexplore_search(ctx=_make_ctx(), query="Nat", limit=1)

    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers.get("authorization") == "Bearer legacy-token"

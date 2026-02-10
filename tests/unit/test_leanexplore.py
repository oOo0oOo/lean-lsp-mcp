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
async def test_leanexplore_search_accepts_upstream_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_urlopen_json(req, timeout: float = 10.0):
        captured["headers"] = dict(req.header_items())
        return {"results": [_sample_item()], "count": 1}

    monkeypatch.delenv("LEAN_EXPLORE_API_KEY", raising=False)
    monkeypatch.setenv("LEANEXPLORE_API_KEY", "alt-token")
    monkeypatch.setattr(server, "_urlopen_json", fake_urlopen_json)

    ctx = _make_ctx()
    await server.leanexplore_search(ctx=ctx, query="Nat", limit=1)

    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers.get("authorization") == "Bearer alt-token"


@pytest.mark.asyncio
async def test_leanexplore_search_summary_extracts_bold_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_urlopen_json(req, timeout: float = 10.0):
        _ = req, timeout
        return {
            "results": [
                {
                    "id": 77,
                    "name": "Nat.add_assoc",
                    "module": "Mathlib.Data.Nat.Basic",
                    "source_text": "theorem Nat.add_assoc : ...",
                    "informalization": "**Associativity.** Addition is associative.",
                }
            ]
        }

    monkeypatch.setattr(server, "_urlopen_json", fake_urlopen_json)

    ctx = _make_ctx()
    result = await server.leanexplore_search_summary(
        ctx=ctx, query="associative", limit=1
    )

    assert len(result.items) == 1
    assert result.items[0].id == 77
    assert result.items[0].lean_name == "Nat.add_assoc"
    assert result.items[0].description == "Associativity."


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
async def test_leanexplore_field_tools_fetch_expected_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_urlopen_json(req, timeout: float = 10.0):
        _ = timeout
        if req.full_url.endswith("/declarations/42"):
            return {
                "id": 42,
                "name": "Nat.add",
                "module": "Mathlib.Data.Nat.Basic",
                "source_text": "theorem Nat.add : Nat -> Nat -> Nat := by ...",
                "source_link": "https://example.test/Nat#L10",
                "docstring": "Addition on natural numbers.",
                "informalization": "Natural number addition.",
                "dependencies": ["Nat.succ", "Nat.zero"],
            }
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr(server, "_urlopen_json", fake_urlopen_json)

    source_result = await server.leanexplore_get_source_code(
        ctx=_make_ctx(), group_id=42
    )
    module_result = await server.leanexplore_get_module(ctx=_make_ctx(), group_id=42)
    docstring_result = await server.leanexplore_get_docstring(
        ctx=_make_ctx(), group_id=42
    )
    link_result = await server.leanexplore_get_source_link(ctx=_make_ctx(), group_id=42)
    deps_result = await server.leanexplore_get_dependencies_field(
        ctx=_make_ctx(), group_id=42
    )

    assert source_result.source_text.startswith("theorem Nat.add")
    assert module_result.module == "Mathlib.Data.Nat.Basic"
    assert docstring_result.docstring == "Addition on natural numbers."
    assert link_result.source_link.endswith("#L10")
    assert deps_result.dependencies == '["Nat.succ","Nat.zero"]'


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
            assert rerank_top == 50
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
        ctx=ctx, query="Nat", package_filters=["Mathlib"], limit=1
    )
    dep_result = await server.leanexplore_dependencies(ctx=ctx, group_id=123, limit=10)

    assert search_result.items[0].id == 123
    assert search_result.items[0].lean_name == "Nat.add"
    assert dep_result.items[0].lean_name == "Nat.succ"

    doc_result = await server.leanexplore_get_docstring(
        ctx=_make_ctx(
            leanexplore_local_enabled=True,
            leanexplore_service=LocalV1Service(),
        ),
        group_id=123,
    )
    assert doc_result.lean_name == "Nat.add"

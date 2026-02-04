from __future__ import annotations

import types

import orjson
import pytest

from lean_lsp_mcp import server


class DummyResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_ctx() -> types.SimpleNamespace:
    lifespan_context = types.SimpleNamespace(rate_limit={"leanexplore": []})
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


def test_leanexplore_search_builds_url_and_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout: float = 10.0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        payload = {
            "results": [_sample_item()],
            "count": 1,
            "total_candidates_considered": 1,
            "processing_time_ms": 5,
        }
        return DummyResponse(orjson.dumps(payload))

    monkeypatch.setenv("LEAN_EXPLORE_BASE_URL", "https://example.test/api/v1")
    monkeypatch.setenv("LEAN_EXPLORE_API_KEY", "token123")
    monkeypatch.setattr(server.urllib.request, "urlopen", fake_urlopen)

    ctx = _make_ctx()
    result = server.leanexplore_search(
        ctx=ctx,
        query="Nat",
        package_filters=["Mathlib", "Batteries"],
        limit=1,
    )

    assert len(result.items) == 1
    assert result.items[0].lean_name == "Nat.add"

    url = captured["url"]
    assert "q=Nat" in url
    assert "pkg=Mathlib" in url
    assert "pkg=Batteries" in url

    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers.get("authorization") == "Bearer token123"


def test_leanexplore_get_by_id_uses_statement_groups_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout: float = 10.0):
        captured["url"] = req.full_url
        return DummyResponse(orjson.dumps(_sample_item()))

    monkeypatch.setenv("LEAN_EXPLORE_BASE_URL", "https://example.test/api/v1")
    monkeypatch.setattr(server.urllib.request, "urlopen", fake_urlopen)

    ctx = _make_ctx()
    result = server.leanexplore_get_by_id(ctx=ctx, group_id=42)

    assert result.id == 42
    assert captured["url"].endswith("/statement_groups/42")

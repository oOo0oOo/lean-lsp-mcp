from __future__ import annotations

import types
from pathlib import Path

import pytest

from lean_lsp_mcp import server


class DummyClient:
    def __init__(self) -> None:
        self.closed_calls = 0

    def close(self) -> None:
        self.closed_calls += 1


class _DiagnosticsClient:
    def __init__(self, responses: list[list[dict]]):
        self._responses = responses
        self.calls: list[float] = []

    def get_diagnostics(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        inactivity_timeout: float = 0.0,
    ) -> list[dict]:
        self.calls.append(inactivity_timeout)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


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


def test_get_diagnostics_with_retry_retries_once() -> None:
    client = _DiagnosticsClient([[], [{"message": "err"}]])
    result = server._get_diagnostics_with_retry(
        client,
        "File.lean",
        start_line=None,
        end_line=None,
        inactivity_timeout=server._DIAGNOSTIC_TIMEOUT_S,
        retry_timeout=server._DIAGNOSTIC_RETRY_TIMEOUT_S,
    )

    assert result == [{"message": "err"}]
    assert client.calls == [server._DIAGNOSTIC_TIMEOUT_S, server._DIAGNOSTIC_RETRY_TIMEOUT_S]


def test_get_diagnostics_with_retry_returns_immediately() -> None:
    client = _DiagnosticsClient([[{"message": "ok"}]])
    result = server._get_diagnostics_with_retry(
        client,
        "File.lean",
        start_line=0,
        end_line=1,
        inactivity_timeout=1.0,
        retry_timeout=5.0,
    )

    assert result == [{"message": "ok"}]
    assert client.calls == [1.0]

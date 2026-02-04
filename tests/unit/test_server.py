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


def test_diagnostic_messages_reopen_on_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_path = tmp_path / "Foo.lean"
    file_path.write_text("theorem foo : True := by trivial\n", encoding="utf-8")

    class DummyResult:
        def __init__(self, diagnostics, success):
            self.diagnostics = diagnostics
            self.success = success

    class DummyClient:
        def __init__(self):
            self.open_calls = []
            self.diag_calls = 0

        def open_file(self, path, dependency_build_mode="never", force_reopen=False):
            self.open_calls.append((dependency_build_mode, force_reopen))

        def get_diagnostics(self, path, start_line=None, end_line=None, inactivity_timeout=15.0):
            self.diag_calls += 1
            if self.diag_calls == 1:
                return DummyResult([], True)
            return DummyResult(
                [
                    {
                        "message": "kernel error",
                        "severity": 1,
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 5},
                        },
                    }
                ],
                False,
            )

    dummy_client = DummyClient()
    ctx = _make_ctx()
    ctx.request_context.lifespan_context.lean_project_path = tmp_path
    ctx.request_context.lifespan_context.client = dummy_client

    monkeypatch.setattr(
        server, "setup_client_for_file", lambda *_args, **_kwargs: "Foo.lean"
    )

    result = server.diagnostic_messages(ctx=ctx, file_path=str(file_path))

    assert len(result.items) == 1
    assert dummy_client.open_calls == [
        ("never", False),
        ("once", True),
    ]

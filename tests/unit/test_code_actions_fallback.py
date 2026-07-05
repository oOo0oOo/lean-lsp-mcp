"""Regression test for the empty-fallback path in `lean_code_actions`.

The fallback fires when the diagnostic-driven scan returns no actions; it
re-queries `get_code_actions` over the full line range. This test mocks
the LSP client so the fallback is observable in isolation — without
relying on a Lean tactic that registers an action-without-diagnostic
(which is rare against current Lean toolchains and was confirmed
unreproducible in the live integration test).
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from lean_lsp_mcp import server as srv


class _Client:
    def __init__(self, code_actions_by_range: dict[tuple, list[dict]]):
        # Map (s_line, s_char, e_line, e_char) -> actions
        self.code_actions_by_range = code_actions_by_range
        self.calls: list[tuple] = []

    def open_file(self, *a, **k):
        pass

    def get_diagnostics(self, *a, **k):
        # No diagnostics on the line — forces the fallback.
        return types.SimpleNamespace(diagnostics=[], success=True, timed_out=False)

    def get_code_actions(self, rel_path, s_line, s_char, e_line, e_char):
        self.calls.append((s_line, s_char, e_line, e_char))
        return self.code_actions_by_range.get((s_line, s_char, e_line, e_char), [])

    def get_code_action_resolve(self, raw):
        return raw


class _Lifespan:
    def __init__(self, client):
        self.client = client
        self.lean_project_path = Path("/tmp")


class _ReqCtx:
    def __init__(self, client):
        self.lifespan_context = _Lifespan(client)
        self.request = None


class _Ctx:
    def __init__(self, client):
        self.request_context = _ReqCtx(client)


@contextmanager
def _fake_lsp_file(client, rel_path):
    client.open_file(rel_path)
    yield types.SimpleNamespace(
        client=client,
        rel_path=rel_path,
        project_path=Path("/tmp"),
        path_policy=types.SimpleNamespace(validate_path=lambda path: path),
    )


@pytest.fixture()
def lean_file(tmp_path: Path) -> Path:
    p = tmp_path / "Fallback.lean"
    p.write_text("import Init\n\nexample : True := by\n  simp?\n")
    return p


def test_fallback_fires_when_no_diagnostic_yet_actions_exist_at_line_range(
    monkeypatch: pytest.MonkeyPatch, lean_file: Path
) -> None:
    """The bug the PR fixes: line has no diagnostic but the line-range
    query DOES yield an action. Pre-PR behaviour would return `[]`.
    """
    # Line 4 is `  simp?`. The fallback's get_code_actions is called with
    # (s_line=3, s_char=0, e_line=3, e_char=len("  simp?")=7) — 0-indexed line.
    range_with_action = (3, 0, 3, 7)
    fake_action = {"title": "Try this: simp", "edit": {"documentChanges": []}}
    client = _Client({range_with_action: [fake_action]})
    monkeypatch.setattr(
        srv, "lsp_client_for_file", lambda _c, p: _fake_lsp_file(client, p)
    )
    monkeypatch.setattr(srv, "resolve_file_path", lambda c, p: lean_file)

    result = srv.code_actions(_Ctx(client), file_path=str(lean_file), line=4)

    assert len(result.actions) == 1
    assert result.actions[0].title == "Try this: simp"
    # The fallback was actually hit: at least one call with s_char=0.
    assert any(call[1] == 0 for call in client.calls), (
        f"fallback never queried full line range: {client.calls}"
    )


def test_fallback_uses_utf16_length_for_end_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The end column passed to `get_code_actions` must be in UTF-16 code
    units, not Python codepoints, so that surrogate-pair characters
    (e.g. `𝕜`) don't undercount the line end. The fallback uses
    `len(line_str.encode("utf-16-le")) // 2`.

    Test fixture: `𝕜 + simp?` — `𝕜` is one Python char but two UTF-16 units.
    """
    f = tmp_path / "Utf16.lean"
    # `𝕜` is U+1D55C (math double-struck capital K); BMP-outside, surrogate pair.
    line_text = "  𝕜 + simp?"  # py-len 11, utf-16 len 12
    f.write_text(f"import Init\n\nexample : True := by\n{line_text}\n")

    expected_end_utf16 = len(line_text.encode("utf-16-le")) // 2
    range_with_action = (3, 0, 3, expected_end_utf16)
    fake_action = {"title": "Try this: trivial", "edit": {"documentChanges": []}}
    client = _Client({range_with_action: [fake_action]})
    monkeypatch.setattr(
        srv, "lsp_client_for_file", lambda _c, p: _fake_lsp_file(client, p)
    )
    monkeypatch.setattr(srv, "resolve_file_path", lambda c, p: f)

    result = srv.code_actions(_Ctx(client), file_path=str(f), line=4)
    assert len(result.actions) == 1, (
        f"expected fallback to use UTF-16 end col {expected_end_utf16}, "
        f"calls were: {client.calls}"
    )
    # The actual end_col used must match the UTF-16 length, not the py-len.
    end_cols_used = [c[3] for c in client.calls if c[1] == 0]
    assert expected_end_utf16 in end_cols_used, (
        f"end_col should equal UTF-16 length {expected_end_utf16}, got {end_cols_used}"
    )


@pytest.mark.parametrize(
    "exc_cls",
    [
        FileNotFoundError,  # `.lake/packages/...` paths
        PermissionError,  # restricted file
        OSError,  # ENAMETOOLONG, EBADF, etc.
        UnicodeDecodeError,  # mojibake content
        ValueError,  # embedded null byte in path
        RuntimeError,  # Path.resolve(strict=True) on a symlink loop
    ],
)
def test_fallback_swallows_resolve_failures_but_logs(
    monkeypatch: pytest.MonkeyPatch, caplog, exc_cls
) -> None:
    """`resolve_file_path` can raise a variety of exceptions depending on
    the failing-path mode: missing file (`FileNotFoundError`), permission
    denied (`PermissionError`), bad path bytes (`OSError`), non-UTF-8
    content (`UnicodeDecodeError`), embedded null bytes (`ValueError`),
    or symlink loops (`RuntimeError` from CPython's ``Path.resolve``).

    Every one must be caught so `lean_code_actions` returns gracefully
    with an empty result + a debug log — never a crash.
    """
    import logging

    client = _Client({})
    monkeypatch.setattr(
        srv, "lsp_client_for_file", lambda _c, p: _fake_lsp_file(client, p)
    )

    def _raise(_c, _p):
        # UnicodeDecodeError requires positional args; construct it carefully.
        if exc_cls is UnicodeDecodeError:
            raise UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1, "simulated")
        raise exc_cls("simulated failure")

    monkeypatch.setattr(srv, "resolve_file_path", _raise)

    with caplog.at_level(logging.DEBUG, logger="lean_lsp_mcp.server"):
        result = srv.code_actions(_Ctx(client), file_path="/no/such/file.lean", line=1)

    assert result.actions == [], (
        f"unexpected actions for {exc_cls.__name__}: {result.actions}"
    )
    assert any("could not read line text" in r.message for r in caplog.records), (
        f"expected debug log for {exc_cls.__name__}; got: {[r.message for r in caplog.records]}"
    )

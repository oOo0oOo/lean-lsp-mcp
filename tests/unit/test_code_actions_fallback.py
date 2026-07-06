"""Regression tests for the empty-fallback path in `lean_code_actions`.

The fallback fires when the diagnostic-driven scan returns no actions; it
re-queries `code_actions` over the full line range. These tests mock the
async LSP client so the fallback is observable in isolation — without
relying on a Lean tactic that registers an action-without-diagnostic
(which is rare against current Lean toolchains and was confirmed
unreproducible in the live integration test).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from leanclient.aio import DiagnosticsReport

from lean_lsp_mcp import server as srv


class _Client:
    def __init__(
        self,
        content: str,
        code_actions_by_range: dict[tuple, list[dict]],
        resolve_exc: Exception | None = None,
    ):
        self._content = content
        self.code_actions_by_range = code_actions_by_range
        self.resolve_exc = resolve_exc
        self.calls: list[tuple] = []

    async def reload_from_disk(self, path: str, wait: bool = False):
        return None

    def content(self, _path: str) -> str:
        return self._content

    async def diagnostics(self, _path: str, fresh: bool = True, timeout=None):
        # No diagnostics on the line — forces the fallback.
        return DiagnosticsReport(items=[], version=1)

    async def code_actions(self, rel_path, s_line, s_char, e_line, e_char, fresh=True):
        self.calls.append((s_line, s_char, e_line, e_char))
        return self.code_actions_by_range.get((s_line, s_char, e_line, e_char), [])

    async def code_action_resolve(self, raw, timeout=30.0):
        if self.resolve_exc is not None:
            raise self.resolve_exc
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


def _patch_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_setup(_ctx, path):
        return path

    monkeypatch.setattr(srv, "setup_client_for_file", fake_setup)


FILE_TEXT = "import Init\n\nexample : True := by\n  simp?\n"


@pytest.mark.asyncio
async def test_fallback_fires_when_no_diagnostic_yet_actions_exist_at_line_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Line has no diagnostic but the line-range query DOES yield an action.
    Pre-fix behaviour would return `[]`.
    """
    # Line 4 is `  simp?`. The fallback queries (3, 0, 3, len("  simp?")=7).
    range_with_action = (3, 0, 3, 7)
    fake_action = {"title": "Try this: simp", "edit": {"documentChanges": []}}
    client = _Client(FILE_TEXT, {range_with_action: [fake_action]})
    _patch_setup(monkeypatch)

    result = await srv.code_actions(_Ctx(client), file_path="Fallback.lean", line=4)

    assert len(result.actions) == 1
    assert result.actions[0].title == "Try this: simp"
    assert any(call[1] == 0 for call in client.calls), (
        f"fallback never queried full line range: {client.calls}"
    )


@pytest.mark.asyncio
async def test_fallback_uses_codepoint_length_for_end_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Columns at the tool layer are codepoints (the async client converts to
    UTF-16 internally), so the fallback's end column is the Python string
    length even on surrogate-pair characters like `𝕜`.
    """
    line_text = "  𝕜 + simp?"  # py-len 11, utf-16 len 12
    text = f"import Init\n\nexample : True := by\n{line_text}\n"

    expected_end = len(line_text)  # codepoints
    range_with_action = (3, 0, 3, expected_end)
    fake_action = {"title": "Try this: trivial", "edit": {"documentChanges": []}}
    client = _Client(text, {range_with_action: [fake_action]})
    _patch_setup(monkeypatch)

    result = await srv.code_actions(_Ctx(client), file_path="Utf16.lean", line=4)
    assert len(result.actions) == 1, (
        f"expected fallback to use codepoint end col {expected_end}, "
        f"calls were: {client.calls}"
    )
    end_cols_used = [c[3] for c in client.calls if c[1] == 0]
    assert expected_end in end_cols_used


@pytest.mark.asyncio
async def test_resolve_failure_skips_action_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An action whose codeAction/resolve fails is skipped, not a crash."""
    range_with_action = (3, 0, 3, 7)
    unresolved_action = {"title": "Try this: simp"}  # no "edit" -> resolve needed
    client = _Client(
        FILE_TEXT,
        {range_with_action: [unresolved_action]},
        resolve_exc=RuntimeError("resolve boom"),
    )
    _patch_setup(monkeypatch)

    result = await srv.code_actions(_Ctx(client), file_path="Fallback.lean", line=4)

    assert result.actions == []

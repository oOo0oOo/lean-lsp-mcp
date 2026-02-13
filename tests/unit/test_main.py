from __future__ import annotations

import sys

import lean_lsp_mcp


def test_main_handles_keyboard_interrupt(monkeypatch) -> None:
    def raise_interrupt(*_args, **_kwargs) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", raise_interrupt)
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp"])

    assert lean_lsp_mcp.main() == 130

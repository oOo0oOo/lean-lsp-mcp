from __future__ import annotations

import os
import sys

import lean_lsp_mcp
import pytest


def test_main_handles_keyboard_interrupt(monkeypatch) -> None:
    def raise_interrupt(*_args, **_kwargs) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", raise_interrupt)
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp"])

    assert lean_lsp_mcp.main() == 130


def test_main_sets_active_transport_env_before_run(monkeypatch) -> None:
    observed: dict[str, str] = {}

    def capture_transport(*_args, **kwargs) -> None:
        observed["transport_arg"] = kwargs["transport"]
        observed["transport_env"] = os.environ.get("LEAN_LSP_MCP_ACTIVE_TRANSPORT", "")

    monkeypatch.delenv("LEAN_LSP_MCP_ACTIVE_TRANSPORT", raising=False)
    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", capture_transport)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lean-lsp-mcp", "--transport", "streamable-http"],
    )

    assert lean_lsp_mcp.main() == 0
    assert observed == {
        "transport_arg": "streamable-http",
        "transport_env": "streamable-http",
    }


def test_main_handles_transport_disconnect_stdio(monkeypatch) -> None:
    def raise_broken_pipe(*_args, **_kwargs) -> None:
        raise BrokenPipeError("broken pipe")

    silenced: list[bool] = []

    def mark_silenced() -> None:
        silenced.append(True)

    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", raise_broken_pipe)
    monkeypatch.setattr(lean_lsp_mcp, "_silence_stdout", mark_silenced)
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp", "--transport", "stdio"])

    assert lean_lsp_mcp.main() == 0
    assert silenced == [True]


class _ExceptionBundle(Exception):
    def __init__(self, *exceptions: BaseException) -> None:
        super().__init__("bundle")
        self.exceptions = exceptions


def test_main_handles_nested_transport_disconnect(monkeypatch) -> None:
    def raise_nested_group(*_args, **_kwargs) -> None:
        raise _ExceptionBundle(
            RuntimeError("outer"),
            _ExceptionBundle(ConnectionResetError("connection closed")),
        )

    silenced: list[bool] = []

    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", raise_nested_group)
    monkeypatch.setattr(lean_lsp_mcp, "_silence_stdout", lambda: silenced.append(True))
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp", "--transport", "stdio"])

    assert lean_lsp_mcp.main() == 0
    assert silenced == [True]


def test_main_does_not_mask_unrelated_exception(monkeypatch) -> None:
    def raise_runtime_error(*_args, **_kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", raise_runtime_error)
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp", "--transport", "stdio"])

    with pytest.raises(RuntimeError, match="boom"):
        lean_lsp_mcp.main()


def test_main_only_swallows_transport_disconnect_for_stdio(monkeypatch) -> None:
    def raise_broken_pipe(*_args, **_kwargs) -> None:
        raise BrokenPipeError("broken pipe")

    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", raise_broken_pipe)
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp", "--transport", "streamable-http"])

    with pytest.raises(BrokenPipeError, match="broken pipe"):
        lean_lsp_mcp.main()

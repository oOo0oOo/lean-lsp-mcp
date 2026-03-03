from __future__ import annotations

import os
import sys
from pathlib import Path

import lean_lsp_mcp
import pytest
from lean_lsp_mcp.coordination import (
    ENV_COORDINATION_DIR,
    ENV_COORDINATION_MODE,
    ENV_INSTANCE_ID,
    ENV_LINEAGE_DEPTH,
    ENV_LINEAGE_ROOT,
    ENV_MAX_LINEAGE_DEPTH,
    ENV_MAX_WORKERS,
)


@pytest.fixture(autouse=True)
def _reset_coordination_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        ENV_COORDINATION_MODE,
        ENV_COORDINATION_DIR,
        ENV_MAX_LINEAGE_DEPTH,
        ENV_MAX_WORKERS,
        ENV_INSTANCE_ID,
        ENV_LINEAGE_ROOT,
        ENV_LINEAGE_DEPTH,
        "LEAN_LSP_MCP_INSTANCE_PID",
    ):
        monkeypatch.delenv(key, raising=False)


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


def test_main_sets_coordination_env_before_run(monkeypatch) -> None:
    observed: dict[str, str] = {}

    def capture_run(*_args, **kwargs) -> None:
        observed["coord_mode"] = os.environ.get(ENV_COORDINATION_MODE, "")
        observed["coord_dir"] = os.environ.get(ENV_COORDINATION_DIR, "")
        observed["max_depth"] = os.environ.get(ENV_MAX_LINEAGE_DEPTH, "")
        observed["max_workers"] = os.environ.get(ENV_MAX_WORKERS, "")
        observed["lineage_root"] = os.environ.get(ENV_LINEAGE_ROOT, "")
        observed["lineage_depth"] = os.environ.get(ENV_LINEAGE_DEPTH, "")
        observed["instance_id"] = os.environ.get(ENV_INSTANCE_ID, "")

    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", capture_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lean-lsp-mcp",
            "--coordination",
            "broker",
            "--coordination-dir",
            "/tmp/lean-lsp-mcp-test",
            "--max-lineage-depth",
            "7",
            "--max-workers",
            "4",
        ],
    )

    assert lean_lsp_mcp.main() == 0
    assert observed["coord_mode"] == "broker"
    assert observed["coord_dir"] == str(Path("/tmp/lean-lsp-mcp-test").resolve())
    assert observed["max_depth"] == "7"
    assert observed["max_workers"] == "4"
    assert observed["lineage_root"]
    assert observed["lineage_depth"] == "0"
    assert observed["instance_id"]


def test_main_normalizes_relative_coordination_dir_from_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, str] = {}

    def capture_run(*_args, **_kwargs) -> None:
        observed["coord_dir"] = os.environ.get(ENV_COORDINATION_DIR, "")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", capture_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lean-lsp-mcp", "--coordination-dir", "coord-relative"],
    )

    assert lean_lsp_mcp.main() == 0
    assert observed["coord_dir"] == str((tmp_path / "coord-relative").resolve())


def test_main_normalizes_relative_coordination_dir_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, str] = {}

    def capture_run(*_args, **_kwargs) -> None:
        observed["coord_dir"] = os.environ.get(ENV_COORDINATION_DIR, "")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV_COORDINATION_DIR, "coord-env-relative")
    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", capture_run)
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp"])

    assert lean_lsp_mcp.main() == 0
    assert observed["coord_dir"] == str((tmp_path / "coord-env-relative").resolve())


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


def test_main_rejects_lineage_depth_over_limit(monkeypatch) -> None:
    monkeypatch.setenv(ENV_LINEAGE_ROOT, "root-id")
    monkeypatch.setenv(ENV_LINEAGE_DEPTH, "9")

    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", lambda **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp", "--max-lineage-depth", "3"])

    with pytest.raises(SystemExit):
        lean_lsp_mcp.main()


def test_main_rejects_non_numeric_inherited_lineage_depth(monkeypatch) -> None:
    monkeypatch.setenv(ENV_LINEAGE_ROOT, "root-id")
    monkeypatch.setenv(ENV_LINEAGE_DEPTH, "oops")
    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", lambda **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp"])

    with pytest.raises(SystemExit):
        lean_lsp_mcp.main()


def test_main_rejects_broker_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(lean_lsp_mcp, "_broker_coordination_supported", lambda: False)
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp", "--coordination", "broker"])

    with pytest.raises(SystemExit):
        lean_lsp_mcp.main()


def test_main_rejects_invalid_coordination_env(monkeypatch) -> None:
    monkeypatch.setenv(ENV_COORDINATION_MODE, "brokr")
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp"])

    with pytest.raises(SystemExit):
        lean_lsp_mcp.main()


def test_main_cli_coordination_overrides_invalid_env(monkeypatch) -> None:
    observed: dict[str, str] = {}

    def capture_run(*_args, **_kwargs) -> None:
        observed["coord_mode"] = os.environ.get(ENV_COORDINATION_MODE, "")

    monkeypatch.setenv(ENV_COORDINATION_MODE, "brokr")
    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", capture_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lean-lsp-mcp", "--coordination", "direct"],
    )

    assert lean_lsp_mcp.main() == 0
    assert observed["coord_mode"] == "direct"


def test_main_cli_max_workers_overrides_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv(ENV_MAX_WORKERS, "abc")
    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", lambda **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp", "--max-workers", "3"])

    assert lean_lsp_mcp.main() == 0
    assert os.environ.get(ENV_MAX_WORKERS) == "3"


def test_main_cli_max_lineage_depth_overrides_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv(ENV_MAX_LINEAGE_DEPTH, "oops")
    monkeypatch.setattr(lean_lsp_mcp.mcp, "run", lambda **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["lean-lsp-mcp", "--max-lineage-depth", "5"])

    assert lean_lsp_mcp.main() == 0
    assert os.environ.get(ENV_MAX_LINEAGE_DEPTH) == "5"

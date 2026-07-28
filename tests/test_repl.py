"""Tests for REPL integration."""

import shutil
from pathlib import Path

import pytest

from lean_lsp_mcp import config
from lean_lsp_mcp import repl as repl_module
from lean_lsp_mcp.repl import (
    Repl,
    ReplProcessError,
    _memory_limit_preexec,
    _split_imports,
    find_repl_binary,
)


# =============================================================================
# Unit Tests
# =============================================================================


def test_find_repl_binary_from_lake_packages(tmp_path: Path, monkeypatch):
    """Auto-detect REPL in .lake/packages."""
    monkeypatch.delenv("LEAN_REPL_PATH", raising=False)
    repl_path = tmp_path / ".lake" / "packages" / "repl" / ".lake" / "build" / "bin"
    repl_path.mkdir(parents=True)
    (repl_path / "repl").touch()

    found = find_repl_binary(str(tmp_path))
    assert found == str(repl_path / "repl")


def test_find_repl_binary_from_uppercase_lake_package(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LEAN_REPL_PATH", raising=False)
    repl_path = tmp_path / ".lake" / "packages" / "REPL" / ".lake" / "build" / "bin"
    repl_path.mkdir(parents=True)
    (repl_path / "repl").touch()

    assert find_repl_binary(str(tmp_path)) == str(repl_path / "repl")


def test_find_repl_binary_env_var_takes_precedence(tmp_path: Path, monkeypatch):
    """LEAN_REPL_PATH env var takes precedence."""
    custom = tmp_path / "custom_repl"
    custom.touch()
    monkeypatch.setenv("LEAN_REPL_PATH", str(custom))

    found = find_repl_binary(str(tmp_path))
    assert found == str(custom)


@pytest.mark.parametrize(
    "code,header,body",
    [
        ("import Mathlib\n\ndef x := 1", "import Mathlib", "def x := 1"),
        (
            "import Foo\nimport Bar\n\ndef x := 1",
            "import Foo\nimport Bar",
            "def x := 1",
        ),
        ("def x := 1", "", "def x := 1"),
        ("import Foo\nimport Foo\n\ndef x := 1", "import Foo", "def x := 1"),
        ("import Foo\n\n\ndef x := 1", "import Foo", "def x := 1"),
    ],
)
def test_split_imports(code: str, header: str, body: str):
    h, b = _split_imports(code)
    assert h == header
    assert b == body


def test_split_imports_preserves_specific_mathlib():
    """Specific Mathlib imports are preserved (faster than import Mathlib)."""
    code = (
        "import Mathlib.Data.Nat\nimport Mathlib.Data.List\nimport Other\n\ndef x := 1"
    )
    header, _ = _split_imports(code)
    assert "import Mathlib.Data.Nat" in header
    assert "import Mathlib.Data.List" in header
    assert "import Other" in header


def test_memory_limit_preexec_uses_configured_mebibytes(monkeypatch):
    calls = []
    monkeypatch.setattr(repl_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        repl_module.resource,
        "setrlimit",
        lambda resource_id, limits: calls.append((resource_id, limits)),
    )

    callback = _memory_limit_preexec(16384)
    assert callback is not None
    callback()

    mem = 16384 * 1024 * 1024
    assert calls == [(repl_module.resource.RLIMIT_AS, (mem, mem))]


@pytest.mark.asyncio
async def test_dead_repl_reports_exit_and_memory_limit(tmp_path: Path):
    class _Stdin:
        def write(self, _data):
            pass

        async def drain(self):
            pass

    class _Stdout:
        async def readline(self):
            return b""

    class _Stderr:
        async def read(self):
            return b"allocation failed"

    class _Proc:
        stdin = _Stdin()
        stdout = _Stdout()
        stderr = _Stderr()
        returncode = -9

        async def wait(self):
            return self.returncode

    repl = Repl(project_dir=str(tmp_path))
    repl.mem_mb = 8192
    repl._proc = _Proc()

    with pytest.raises(ReplProcessError) as exc_info:
        await repl._send_cmd("#check Nat")

    message = str(exc_info.value)
    assert "exit code -9" in message
    assert "allocation failed" in message
    assert "8192 MiB (LEAN_REPL_MEM_MB)" in message


@pytest.mark.asyncio
async def test_run_code_uses_cached_header_environment(tmp_path: Path):
    from unittest.mock import AsyncMock

    repl = Repl(project_dir=str(tmp_path))
    repl._ensure_header = AsyncMock(return_value=7)
    repl._send_cmd = AsyncMock(
        return_value={
            "messages": [
                {
                    "severity": "info",
                    "data": "Nat : Type",
                    "pos": {"line": 1, "column": 0},
                }
            ]
        }
    )

    result = await repl.run_code("import Mathlib\n\n#check Nat")

    repl._ensure_header.assert_awaited_once_with("import Mathlib")
    repl._send_cmd.assert_awaited_once_with("#check Nat", env=7)
    assert result.line_offset == 2
    assert result.messages[0]["data"] == "Nat : Type"


@pytest.mark.asyncio
async def test_run_snippets_uses_last_sorry(tmp_path: Path):
    """When body already contains sorries, the *last* (injected) one is used."""
    from unittest.mock import AsyncMock

    repl = Repl(project_dir=str(tmp_path))
    repl._proc = True  # fake "running"
    repl._header = ""
    repl._header_env = 0

    # Simulate a response with two sorries: pre-existing (ps=10) + injected (ps=42)
    cmd_resp = {
        "sorries": [
            {"proofState": 10},  # pre-existing sorry
            {"proofState": 42},  # our injected sorry
        ]
    }
    tactic_resp = {"goals": ["⊢ True"]}

    repl._ensure_header = AsyncMock(return_value=0)
    repl._send_cmd = AsyncMock(return_value=cmd_resp)
    repl._send_tactic = AsyncMock(return_value=tactic_resp)

    results = await repl.run_snippets(
        "import Mathlib\n\ntheorem t : True := by\n  have : False := sorry",
        ["trivial"],
    )

    # Should have used proofState 42 (last sorry), not 10
    repl._send_tactic.assert_called_once_with("trivial", 42)
    assert results[0].goals == ["⊢ True"]


@pytest.mark.asyncio
async def test_run_snippets_matches_body_indentation(tmp_path: Path):
    """Injected sorry matches the indentation of surrounding tactic lines."""
    from unittest.mock import AsyncMock

    repl = Repl(project_dir=str(tmp_path))
    repl._proc = True
    repl._header = ""
    repl._header_env = 0

    cmd_resp = {"sorries": [{"proofState": 1}]}
    tactic_resp = {"goals": []}

    repl._ensure_header = AsyncMock(return_value=0)
    repl._send_cmd = AsyncMock(return_value=cmd_resp)
    repl._send_tactic = AsyncMock(return_value=tactic_resp)

    # Body has 4-space indented tactics
    await repl.run_snippets(
        "import Foo\n\ntheorem t : True := by\n    intro h",
        ["trivial"],
    )

    # Verify the sorry was appended with 4-space indent, not hard-coded 2
    sent_code = repl._send_cmd.call_args[0][0]
    assert sent_code.endswith("    sorry"), (
        f"Expected 4-space indent, got: {sent_code!r}"
    )


# =============================================================================
# Integration Tests (require REPL binary)
# =============================================================================


@pytest.fixture
async def repl(test_project_path: Path, monkeypatch):
    # Check multiple possible locations for REPL binary
    candidates = [
        test_project_path / ".lake" / "build" / "bin" / "repl",
        test_project_path
        / ".lake"
        / "packages"
        / "repl"
        / ".lake"
        / "build"
        / "bin"
        / "repl",
        test_project_path
        / ".lake"
        / "packages"
        / "REPL"
        / ".lake"
        / "build"
        / "bin"
        / "repl",
    ]
    repl_bin = next((p for p in candidates if p.exists()), None)
    if not repl_bin:
        found = shutil.which("repl")
        repl_bin = Path(found) if found else None
    if not repl_bin:
        pytest.skip("REPL binary not found")

    monkeypatch.setenv("LEAN_REPL_PATH", str(repl_bin))
    monkeypatch.setenv("LEAN_REPL_TIMEOUT", "30")
    monkeypatch.delenv(config.REPL_MEM_MB_ENV, raising=False)

    r = Repl(project_dir=str(test_project_path))
    yield r
    await r.close()


@pytest.mark.asyncio
async def test_run_snippets_returns_results(repl: Repl):
    results = await repl.run_snippets("theorem t : 1 = 1 := by\n", ["rfl", "simp"])
    assert len(results) == 2
    assert any(r.error is None for r in results)


@pytest.mark.asyncio
async def test_header_caching_reuses_repl(repl: Repl):
    """Header caching keeps same REPL across calls with same imports."""
    base = "theorem t : 1 = 1 := by\n"

    await repl.run_snippets(base, ["rfl"])
    proc1 = repl._proc

    await repl.run_snippets(base, ["rfl"])
    proc2 = repl._proc

    assert proc1 is proc2  # Same process reused


@pytest.mark.asyncio
async def test_backtracking_isolation(repl: Repl):
    base = "theorem t : 1 + 1 = 2 := by\n"
    results = await repl.run_snippets(
        base,
        [
            "have h : False := sorry; exact h.elim",
            "rfl",
        ],
    )
    assert len(results) == 2
    assert results[1].error is None  # rfl specifically should work


@pytest.mark.asyncio
async def test_default_memory_limit_keeps_mathlib_repl_alive(repl: Repl):
    """The default cap must leave enough virtual address space to load Mathlib."""
    assert repl.mem_mb == 16384

    result = await repl.run_code("import Mathlib\n\n#check Nat")
    invalid = await repl.run_code("import Mathlib\n\nexample : 1 = 2 := by rfl")
    first_definition = await repl.run_code("import Mathlib\n\ndef replRunValue := 1")
    repeated_definition = await repl.run_code("import Mathlib\n\ndef replRunValue := 1")

    assert result.error is None
    assert any("Nat" in message.get("data", "") for message in result.messages)
    assert any(message.get("severity") == "error" for message in invalid.messages)
    assert not any(
        message.get("severity") == "error" for message in first_definition.messages
    )
    assert not any(
        message.get("severity") == "error" for message in repeated_definition.messages
    )
    assert repl._proc is not None
    assert repl._proc.returncode is None


@pytest.mark.skip
@pytest.mark.asyncio
async def test_benchmark_repl_vs_lsp(repl: Repl, mcp_client_factory, test_project_path):
    """Benchmark: REPL vs LSP for multi_attempt.

    Uses GoalSample.lean which has `import Mathlib` - same as most test files.
    REPL advantage: tactic mode runs tactics without reparsing body.
    """
    import json
    import time

    from tests.helpers.mcp_client import result_text

    tactics = ["rfl", "simp", "omega", "decide", "trivial", "rfl", "simp", "rfl"]
    # Same as BenchmarkTest.lean - fair comparison with LSP
    # Goal: 1 = 1 (solvable by rfl, simp, omega, decide)
    base = """\
import Mathlib

theorem benchmark : 1 = 1 := by
  """

    # Cold REPL (starts subprocess + loads Mathlib + first body parse)
    start = time.perf_counter()
    await repl.run_snippets(base, tactics)
    repl_cold = time.perf_counter() - start

    # Warm REPL (subprocess cached with Mathlib, body re-parsed but tactics fast)
    start = time.perf_counter()
    repl_results = await repl.run_snippets(base, tactics)
    repl_warm = time.perf_counter() - start

    # Cold LSP - use BenchmarkTest.lean which has same `import Mathlib`
    # Line 6 is `sorry` - multi_attempt replaces it with each tactic
    async with mcp_client_factory() as client:
        start = time.perf_counter()
        await client.call_tool(
            "lean_multi_attempt",
            {
                "file_path": str(test_project_path / "BenchmarkTest.lean"),
                "line": 6,
                "snippets": tactics,
            },
        )
        lsp_cold = time.perf_counter() - start

        # Warm LSP (file compiled, but each tactic is separate edit)
        start = time.perf_counter()
        lsp_result = await client.call_tool(
            "lean_multi_attempt",
            {
                "file_path": str(test_project_path / "BenchmarkTest.lean"),
                "line": 6,
                "snippets": tactics,
            },
        )
        lsp_warm = time.perf_counter() - start

    lsp_data = json.loads(result_text(lsp_result))
    repl_success = [r.goals == [] and r.error is None for r in repl_results]
    lsp_success = [a["goals"] == [] for a in lsp_data["items"]]

    # Per-tactic timing
    repl_per = repl_warm / len(tactics) * 1000
    lsp_per = lsp_warm / len(tactics) * 1000

    print(f"\n{'=' * 60}")
    print(f"Tactics: {len(tactics)}")
    print(
        f"REPL cold: {repl_cold:.2f}s | warm: {repl_warm * 1000:.0f}ms ({repl_per:.0f}ms/tactic)"
    )
    print(
        f"LSP  cold: {lsp_cold:.2f}s | warm: {lsp_warm * 1000:.0f}ms ({lsp_per:.0f}ms/tactic)"
    )
    if repl_warm < lsp_warm:
        print(f"REPL is {lsp_warm / repl_warm:.1f}x faster (warm)")
    else:
        print(f"LSP is {repl_warm / lsp_warm:.1f}x faster (warm)")
    print(f"{'=' * 60}")

    assert repl_success == lsp_success, "Results differ!"

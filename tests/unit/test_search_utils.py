import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def reload_search_utils():
    # Ensure a clean module state for each test once the module exists.
    import lean_lsp_mcp.search_utils as search_utils

    importlib.reload(search_utils)
    return search_utils


def test_check_ripgrep_status_when_rg_available(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    monkeypatch.setattr(search_utils.shutil, "which", lambda _: "/usr/bin/rg")

    available, message = search_utils.check_ripgrep_status()

    assert available is True
    assert message == ""


@pytest.mark.parametrize(
    "platform_name, expected_snippets",
    [
        (
            "Windows",
            [
                "winget install BurntSushi.ripgrep.MSVC",
                "choco install ripgrep",
            ],
        ),
        (
            "Darwin",
            [
                "brew install ripgrep",
            ],
        ),
        (
            "Linux",
            [
                "sudo apt-get install ripgrep",
                "sudo dnf install ripgrep",
            ],
        ),
        (
            "FreeBSD",
            [
                "Check alternative installation methods.",
            ],
        ),
    ],
)
def test_check_ripgrep_status_when_rg_missing_platform_specific(
    monkeypatch, reload_search_utils, platform_name, expected_snippets
):
    search_utils = reload_search_utils

    monkeypatch.setattr(search_utils.shutil, "which", lambda _: None)
    monkeypatch.setattr(search_utils.platform, "system", lambda: platform_name)

    available, message = search_utils.check_ripgrep_status()

    assert available is False
    assert "ripgrep (rg) was not found on your PATH" in message
    assert "https://github.com/BurntSushi/ripgrep#installation" in message

    for snippet in expected_snippets:
        assert snippet in message


def _make_match(path: str, line: str) -> str:
    return json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": path},
                "lines": {"text": line},
            },
        }
    )


class _DummyCompletedProcess:
    def __init__(self, stdout_lines, returncode=0, stderr_text=""):
        self.stdout = "".join(f"{line}\n" for line in stdout_lines)
        self.stderr = stderr_text
        self.returncode = returncode
        self.args = []


def _configure_env(monkeypatch, search_utils, stdout_events, returncode=0, expected_cwd=None):
    completed = _DummyCompletedProcess(stdout_events, returncode=returncode)

    def fake_check():
        return True, ""

    run_calls = []

    def fake_run(cmd, *, capture_output=False, text=False, cwd=None):
        run_calls.append((cmd, cwd))
        if expected_cwd is not None:
            assert cwd == expected_cwd
        return completed

    monkeypatch.setattr(search_utils, "check_ripgrep_status", fake_check)
    monkeypatch.setattr(search_utils.subprocess, "run", fake_run)

    return completed, run_calls


def test_lean_search_returns_matching_results(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match("src/Foo/Bar.lean", "def target : Nat := 0"),
        _make_match("src/Foo/Baz.lean", "lemma target : True := by trivial"),
    ]

    _configure_env(
        monkeypatch,
        search_utils,
        events,
        expected_cwd=str(project_root.resolve()),
    )

    results = search_utils.lean_search("target", project_root=project_root)

    assert results == [
        {
            "name": "target",
            "kind": "def",
            "file": "src/Foo/Bar.lean",
        },
        {
            "name": "target",
            "kind": "lemma",
            "file": "src/Foo/Baz.lean",
        },
    ]


def test_lean_search_exact_match(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match("src/Foo/Bar.lean", "def sampleValue : Nat := 0"),
        _make_match("src/Foo/Bar.lean", "def sampleValueExtra : Nat := 0"),
    ]

    _configure_env(
        monkeypatch,
        search_utils,
        events,
        expected_cwd=str(project_root.resolve()),
    )

    results = search_utils.lean_search("sampleValue", project_root=project_root)

    assert results == [
        {
            "name": "sampleValue",
            "kind": "def",
            "file": "src/Foo/Bar.lean",
        },
        {
            "name": "sampleValueExtra",
            "kind": "def",
            "file": "src/Foo/Bar.lean",
        },
    ]


def test_lean_search_respects_limit(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match("src/Foo/Bar.lean", "def dup : Nat := 0"),
        _make_match("src/Foo/Baz.lean", "def dup : Nat := 0"),
        _make_match("src/Foo/Qux.lean", "def dup : Nat := 0"),
    ]

    _configure_env(
        monkeypatch,
        search_utils,
        events,
        expected_cwd=str(project_root.resolve()),
    )

    results = search_utils.lean_search("dup", limit=2, project_root=project_root)

    assert len(results) == 2


def test_lean_search_returns_relative_paths(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match(
            ".lake/packages/mathlib/Mathlib/Algebra/Group.lean",
            "theorem sampleGroupTheorem : True := by trivial",
        )
    ]

    _configure_env(
        monkeypatch,
        search_utils,
        events,
        expected_cwd=str(project_root.resolve()),
    )

    results = search_utils.lean_search("sampleGroupTheorem", project_root=project_root)

    assert results == [
        {
            "name": "sampleGroupTheorem",
            "kind": "theorem",
            "file": ".lake/packages/mathlib/Mathlib/Algebra/Group.lean",
        }
    ]


def test_lean_search_handles_ripgrep_errors(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    project_root = Path("/proj")
    _configure_env(
        monkeypatch,
        search_utils,
        [],
        returncode=2,
        expected_cwd=str(project_root.resolve()),
    )

    with pytest.raises(RuntimeError):
        search_utils.lean_search("sample", project_root=project_root)


def test_lean_search_returns_empty_for_no_matches(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    project_root = Path("/proj")

    _configure_env(
        monkeypatch,
        search_utils,
        [],
        expected_cwd=str(project_root.resolve()),
    )

    assert search_utils.lean_search("nothing", project_root=project_root) == []


TEST_PROJECT_ROOT = Path(__file__).resolve().parents[1] / "test_project"


def test_lean_search_integration_project_root(reload_search_utils):
    search_utils = reload_search_utils
    available, message = search_utils.check_ripgrep_status()
    if not available:
        pytest.skip(message)

    results = search_utils.lean_search("sampleTheorem", project_root=TEST_PROJECT_ROOT)

    assert results == [
        {
            "name": "sampleTheorem",
            "kind": "theorem",
            "file": "EditorTools.lean",
        }
    ]


def test_lean_search_integration_mathlib(reload_search_utils):
    search_utils = reload_search_utils
    available, message = search_utils.check_ripgrep_status()
    if not available:
        pytest.skip(message)

    results = search_utils.lean_search(
        "map_mul_right",
        limit=5,
        project_root=TEST_PROJECT_ROOT,
    )

    assert results
    assert any(
        item == {
            "name": "map_mul_right",
            "kind": "theorem",
            "file": ".lake/packages/mathlib/Mathlib/GroupTheory/MonoidLocalization/Basic.lean",
        }
        for item in results
    )


def test_lean_search_integration_mathlib_prefix_results(reload_search_utils):
    search_utils = reload_search_utils
    available, message = search_utils.check_ripgrep_status()
    if not available:
        pytest.skip(message)

    results = search_utils.lean_search(
        "add_comm",
        limit=5,
        project_root=TEST_PROJECT_ROOT,
    )

    assert len(results) >= 2
    assert any(
        item == {
            "name": "add_comm_zero",
            "kind": "theorem",
            "file": ".lake/packages/mathlib/MathlibTest/Find.lean",
        }
        for item in results
    )


def test_lean_search_integration_mathlib_prefix_limit(reload_search_utils):
    search_utils = reload_search_utils
    available, message = search_utils.check_ripgrep_status()
    if not available:
        pytest.skip(message)

    results = search_utils.lean_search(
        "add_comm",
        limit=1,
        project_root=TEST_PROJECT_ROOT,
    )

    assert len(results) == 1
    assert results[0]["name"].startswith("add_comm")

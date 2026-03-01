import importlib
import io
import orjson
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
    return orjson.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": path},
                "lines": {"text": line},
            },
        }
    ).decode("utf-8")


class _DummyCompletedProcess:
    def __init__(self, stdout_lines, returncode=0, stderr_text=""):
        self.stdout = "".join(f"{line}\n" for line in stdout_lines)
        self.stderr = stderr_text
        self.returncode = returncode
        self.args = []


class _DummyPopen:
    def __init__(self, stdout_lines, returncode=0, stderr_text=""):
        self.stdout = io.StringIO("".join(f"{line}\n" for line in stdout_lines))
        self.stderr = io.StringIO(stderr_text)
        self.returncode = None
        self._final_code = returncode

    def wait(self, timeout=None):
        self.returncode = self._final_code
        return self._final_code

    def terminate(self):
        self.returncode = self._final_code

    def kill(self):
        self.returncode = self._final_code


def _configure_env(
    monkeypatch, search_utils, stdout_events, returncode=0, expected_cwd=None
):
    lean_completed = _DummyCompletedProcess(["/nonexistent/lean"], returncode=0)

    def fake_check():
        return True, ""

    run_calls = []
    create_calls = []

    def fake_create(cmd, *, cwd):
        create_calls.append((cmd, cwd))
        if expected_cwd is not None and cmd and cmd[0] == "rg":
            assert cwd == expected_cwd
        return _DummyPopen(stdout_events, returncode=returncode)

    def fake_run(cmd, *, capture_output=False, text=False, cwd=None):
        run_calls.append((cmd, cwd))
        if cmd[:2] == ["lean", "--print-prefix"]:
            return lean_completed
        raise AssertionError(f"Unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(search_utils, "check_ripgrep_status", fake_check)
    monkeypatch.setattr(search_utils, "_create_ripgrep_process", fake_create)
    monkeypatch.setattr(search_utils.subprocess, "run", fake_run)

    return create_calls, run_calls


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

    results = search_utils.lean_local_search("target", project_root=project_root)

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

    results = search_utils.lean_local_search("sampleValue", project_root=project_root)

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


def test_lean_search_prioritizes_exact_and_local_results(
    monkeypatch, reload_search_utils
):
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match(
            ".lake/packages/mathlib/Mathlib/Foo.lean",
            "def sampleValue : Nat := 0",
        ),
        _make_match("src/Foo/Bar.lean", "def sampleValueExtra : Nat := 0"),
        _make_match("src/Foo/Baz.lean", "def sampleValue : Nat := 1"),
    ]

    _configure_env(
        monkeypatch,
        search_utils,
        events,
        expected_cwd=str(project_root.resolve()),
    )

    results = search_utils.lean_local_search(
        "sampleValue", limit=3, project_root=project_root
    )

    assert results == [
        {
            "name": "sampleValue",
            "kind": "def",
            "file": "src/Foo/Baz.lean",
        },
        {
            "name": "sampleValue",
            "kind": "def",
            "file": ".lake/packages/mathlib/Mathlib/Foo.lean",
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

    results = search_utils.lean_local_search("dup", limit=2, project_root=project_root)

    assert len(results) == 2


def test_lean_search_oversamples_before_ranking(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match("src/Foo/A.lean", "def sampleValueExtraA : Nat := 0"),
        _make_match("src/Foo/B.lean", "def sampleValueExtraB : Nat := 0"),
        _make_match("src/Foo/C.lean", "def sampleValueExtraC : Nat := 0"),
        _make_match("src/Foo/D.lean", "def sampleValue : Nat := 0"),
    ]

    _configure_env(
        monkeypatch,
        search_utils,
        events,
        expected_cwd=str(project_root.resolve()),
    )

    results = search_utils.lean_local_search(
        "sampleValue", limit=1, project_root=project_root
    )

    assert results == [
        {
            "name": "sampleValue",
            "kind": "def",
            "file": "src/Foo/D.lean",
        }
    ]


def test_lean_search_ranks_dotted_queries_by_full_name(
    monkeypatch, reload_search_utils
):
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match("src/Foo/A.lean", "def Nat.succX : Nat := 0"),
        _make_match("src/Foo/B.lean", "def Nat.succ : Nat := 1"),
        _make_match("src/Foo/C.lean", "def Prelude.Nat.succExtra : Nat := 2"),
    ]

    _configure_env(
        monkeypatch,
        search_utils,
        events,
        expected_cwd=str(project_root.resolve()),
    )

    results = search_utils.lean_local_search(
        "Nat.succ", limit=3, project_root=project_root
    )

    assert results[0] == {
        "name": "Nat.succ",
        "kind": "def",
        "file": "src/Foo/B.lean",
    }


def test_lean_search_deduplicates_identical_results(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match("src/Foo/Bar.lean", "def dupName : Nat := 0"),
        _make_match("src/Foo/Bar.lean", "def dupName : Nat := 0"),
        _make_match("src/Foo/Baz.lean", "def dupNameExtra : Nat := 0"),
    ]

    _configure_env(
        monkeypatch,
        search_utils,
        events,
        expected_cwd=str(project_root.resolve()),
    )

    results = search_utils.lean_local_search("dupName", project_root=project_root)

    assert results == [
        {
            "name": "dupName",
            "kind": "def",
            "file": "src/Foo/Bar.lean",
        },
        {
            "name": "dupNameExtra",
            "kind": "def",
            "file": "src/Foo/Baz.lean",
        },
    ]


def test_lean_search_wait_timeout_only_on_early_termination(
    monkeypatch, reload_search_utils
):
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match(f"src/Foo/{idx}.lean", f"def my{idx} : Nat := {idx}")
        for idx in range(10)
    ]

    waits: list[float | None] = []
    terminate_calls: list[None] = []

    class _RecordingPopen(_DummyPopen):
        def wait(self, timeout=None):
            waits.append(timeout)
            return super().wait(timeout=timeout)

        def terminate(self):
            terminate_calls.append(None)
            return super().terminate()

    monkeypatch.setattr(
        search_utils, "_get_lean_src_search_path", lambda _root=None: None
    )
    monkeypatch.setattr(
        search_utils,
        "_create_ripgrep_process",
        lambda cmd, *, cwd: _RecordingPopen(events),
    )

    search_utils.lean_local_search("my", limit=10, project_root=project_root)
    assert waits == [None]

    waits.clear()
    search_utils.lean_local_search("my", limit=1, project_root=project_root)
    assert terminate_calls
    assert waits == [5]


def test_lean_search_reaps_process_on_parse_error(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [_make_match("src/Foo/Bar.lean", "def ok : Nat := 0")]

    calls: dict[str, int] = {"terminate": 0, "kill": 0, "wait": 0}

    class _RecordingPopen(_DummyPopen):
        def wait(self, timeout=None):
            calls["wait"] += 1
            return super().wait(timeout=timeout)

        def terminate(self):
            calls["terminate"] += 1
            return super().terminate()

        def kill(self):
            calls["kill"] += 1
            return super().kill()

    monkeypatch.setattr(
        search_utils, "_get_lean_src_search_path", lambda _root=None: None
    )
    monkeypatch.setattr(
        search_utils,
        "_create_ripgrep_process",
        lambda cmd, *, cwd: _RecordingPopen(events),
    )

    def _raise_on_load(_: str):
        raise ValueError("bad json")

    monkeypatch.setattr(search_utils, "_json_loads", _raise_on_load)

    with pytest.raises(ValueError):
        search_utils.lean_local_search("ok", project_root=project_root)

    assert calls["wait"] >= 1
    assert calls["terminate"] + calls["kill"] >= 1


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

    results = search_utils.lean_local_search(
        "sampleGroupTheorem", project_root=project_root
    )

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

    # With exit code 2 and no results, should raise RuntimeError
    with pytest.raises(RuntimeError):
        search_utils.lean_local_search("sample", project_root=project_root)


def test_lean_search_handles_ripgrep_errors_with_partial_results(
    monkeypatch, reload_search_utils
):
    """Test that partial results are returned even when ripgrep exits with error code 2."""
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match("src/Test.lean", "def partialResult : Nat := 0"),
    ]
    _configure_env(
        monkeypatch,
        search_utils,
        events,
        returncode=2,  # Error after getting some results
        expected_cwd=str(project_root.resolve()),
    )

    # Should return partial results instead of raising
    results = search_utils.lean_local_search(
        "partial", project_root=project_root, limit=10
    )
    assert len(results) == 1
    assert results[0]["name"] == "partialResult"


def test_lean_search_returns_empty_for_no_matches(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    project_root = Path("/proj")

    _configure_env(
        monkeypatch,
        search_utils,
        [],
        expected_cwd=str(project_root.resolve()),
    )

    assert search_utils.lean_local_search("nothing", project_root=project_root) == []


TEST_PROJECT_ROOT = Path(__file__).resolve().parents[1] / "test_project"


def test_lean_search_integration_project_root(reload_search_utils):
    search_utils = reload_search_utils
    available, message = search_utils.check_ripgrep_status()
    if not available:
        pytest.skip(message)

    results = search_utils.lean_local_search(
        "sampleTheorem", project_root=TEST_PROJECT_ROOT
    )

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

    results = search_utils.lean_local_search(
        "map_mul_right",
        limit=50,
        project_root=TEST_PROJECT_ROOT,
    )

    assert results
    assert any(
        item
        == {
            "name": "map_mul_right",
            "kind": "theorem",
            "file": ".lake/packages/mathlib/Mathlib/GroupTheory/MonoidLocalization/Maps.lean",
        }
        for item in results
    )


def test_lean_search_integration_mathlib_prefix_results(reload_search_utils):
    search_utils = reload_search_utils
    available, message = search_utils.check_ripgrep_status()
    if not available:
        pytest.skip(message)

    results = search_utils.lean_local_search(
        "add_comm",
        limit=50,
        project_root=TEST_PROJECT_ROOT,
    )

    assert len(results) >= 2
    assert any(
        item
        == {
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

    results = search_utils.lean_local_search(
        "add_comm",
        limit=1,
        project_root=TEST_PROJECT_ROOT,
    )

    assert len(results) == 1
    assert results[0]["name"].split(".")[-1].startswith("add_comm")


def test_lean_search_integration_stdlib_definitions(reload_search_utils):
    search_utils = reload_search_utils
    available, message = search_utils.check_ripgrep_status()
    if not available:
        pytest.skip(message)

    results = search_utils.lean_local_search(
        "Nat.succ",
        limit=20,
        project_root=TEST_PROJECT_ROOT,
    )

    assert results
    assert any(item["name"].startswith("Nat.succ") for item in results)


def test_lean_search_all_declaration_types(monkeypatch, reload_search_utils):
    """Test all supported declaration types are correctly parsed."""
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match("Test.lean", "theorem myTheorem : True := by trivial"),
        _make_match("Test.lean", "lemma myLemma : True := by trivial"),
        _make_match("Test.lean", "def myDef : Nat := 42"),
        _make_match("Test.lean", "axiom myAxiom : Prop"),
        _make_match("Test.lean", "class MyClass (α : Type) where"),
        _make_match("Test.lean", "instance myInstance : MyClass Nat where"),
        _make_match("Test.lean", "structure MyStruct where"),
        _make_match("Test.lean", "inductive MyInductive where"),
        _make_match("Test.lean", "abbrev MyAbbrev := Nat"),
        _make_match("Test.lean", "opaque myOpaque : Nat"),
    ]

    _configure_env(
        monkeypatch, search_utils, events, expected_cwd=str(project_root.resolve())
    )
    results = search_utils.lean_local_search("my", project_root=project_root)

    assert len(results) == 10
    assert {r["kind"] for r in results} == {
        "theorem",
        "lemma",
        "def",
        "axiom",
        "class",
        "instance",
        "structure",
        "inductive",
        "abbrev",
        "opaque",
    }


def test_lean_search_strips_colon_from_names(monkeypatch, reload_search_utils):
    """Test that declaration names with colons are correctly stripped."""
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match("Test.lean", "def myFunc: Nat := 42"),
        _make_match("Test.lean", "theorem myThm : True := by trivial"),
    ]

    _configure_env(
        monkeypatch, search_utils, events, expected_cwd=str(project_root.resolve())
    )
    results = search_utils.lean_local_search("my", project_root=project_root)

    assert len(results) == 2
    assert {result["name"] for result in results} == {"myFunc", "myThm"}
    assert all(":" not in result["name"] for result in results)


def test_lean_search_uses_cwd_when_project_root_none(monkeypatch, reload_search_utils):
    """Test that Path.cwd() is used when project_root is None."""
    search_utils = reload_search_utils
    fake_cwd = Path("/fake/working/dir")
    monkeypatch.setattr(Path, "cwd", lambda: fake_cwd)

    events = [_make_match("Test.lean", "def testDef : Nat := 0")]
    _configure_env(
        monkeypatch, search_utils, events, expected_cwd=str(fake_cwd.resolve())
    )

    results = search_utils.lean_local_search("testDef", project_root=None)

    assert len(results) == 1
    assert results[0]["name"] == "testDef"


def test_lean_search_resolves_project_root_to_absolute(
    monkeypatch, reload_search_utils
):
    """Test that project_root is resolved to an absolute path."""
    search_utils = reload_search_utils
    absolute_root = Path("/absolute/path/to/project")
    events = [_make_match("Test.lean", "def testFunc : Nat := 0")]

    _configure_env(
        monkeypatch,
        search_utils,
        events,
        expected_cwd=str(absolute_root.resolve()),
    )
    results = search_utils.lean_local_search("testFunc", project_root=absolute_root)

    assert len(results) == 1
    assert results[0]["name"] == "testFunc"


def test_lean_search_uses_project_root_not_cwd(monkeypatch, reload_search_utils):
    """Test that project_root is used instead of cwd when both differ."""
    search_utils = reload_search_utils
    current_dir = Path("/home/user/workspace")
    project_root = Path("/different/project")

    monkeypatch.setattr(Path, "cwd", lambda: current_dir)
    events = [_make_match("MyModule.lean", "def myDef : Nat := 42")]

    _configure_env(
        monkeypatch,
        search_utils,
        events,
        expected_cwd=str(project_root.resolve()),
    )
    results = search_utils.lean_local_search("myDef", project_root=project_root)

    assert len(results) == 1
    assert results[0]["file"] == "MyModule.lean"


def test_lean_search_handles_namespaces(monkeypatch, reload_search_utils):
    """Test that searching without namespace prefix finds namespaced declarations."""
    search_utils = reload_search_utils
    project_root = Path("/proj")
    events = [
        _make_match("Nat.lean", "def Nat.add (a b : Nat) : Nat := a + b"),
        _make_match("List.lean", "def List.add (xs ys : List α) : List α := xs ++ ys"),
        _make_match("Basic.lean", "def add (x y : Int) : Int := x + y"),
    ]

    _configure_env(
        monkeypatch, search_utils, events, expected_cwd=str(project_root.resolve())
    )
    results = search_utils.lean_local_search("add", project_root=project_root)

    assert len(results) == 3
    names = {r["name"] for r in results}
    assert names == {"Nat.add", "List.add", "add"}

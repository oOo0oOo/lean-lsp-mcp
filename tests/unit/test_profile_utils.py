"""Unit tests for profile_utils module."""

import os
import platform
from pathlib import Path

import pytest

from lean_lsp_mcp import profile_utils
from lean_lsp_mcp.profile_utils import (
    _build_proof_items,
    _extract_line_times,
    _extract_theorem_source,
    _filter_categories,
    _parse_output,
    _run_lean_profile,
    profile_theorem,
)

# Status markers exactly as `lean --profile` emits them, captured from a real
# run rather than written by hand: success, failure, error.
OK = "\u2705\ufe0f"
FAIL = "\u274c\ufe0f"
ERROR = "\U0001f4a5\ufe0f"

# Most extraction tests give the file a header, so that they do not depend on
# the layout of a file without one (tested on its own).
HEADER = ["import Lean", ""]


def _theorem_lines(lines: list[str], target_line: int) -> list[str]:
    """The extracted snippet, from the target theorem's own line on."""
    source, _, theorem_start = _extract_theorem_source(lines, target_line)
    return source.splitlines()[theorem_start - 1 :]


class TestParseOutput:
    def test_parses_single_trace(self):
        output = "[Elab.command] [0.005] some message\n"
        traces, _ = _parse_output(output)
        assert traces == [(0, "Elab.command", 5.0, "some message")]

    def test_parses_indented_traces(self):
        output = "[A] [0.001] outer\n  [B] [0.002] inner\n"
        traces, _ = _parse_output(output)
        assert [(d, c) for d, c, _, _ in traces] == [(0, "A"), (1, "B")]

    def test_parses_cumulative_ms_and_s(self):
        output = "cumulative profiling times:\n        simp 2ms\n        import 4.5s\n"
        _, cumulative = _parse_output(output)
        assert cumulative == {"simp": 2.0, "import": 4500.0}

    def test_ignores_non_trace_lines(self):
        output = "random text\n[A] [0.001] msg\nmore text\n"
        traces, _ = _parse_output(output)
        assert len(traces) == 1


class TestBuildProofItems:
    def test_builds_items_from_proof_lines(self):
        source = ["theorem foo := by", "  simp", "  · ring", ""]
        items = _build_proof_items(source, 1)
        # (line_no, content, is_bullet)
        assert items == [(2, "simp", False), (3, "ring", True)]

    def test_skips_comments(self):
        source = ["by", "  -- comment", "  simp"]
        items = _build_proof_items(source, 1)
        assert items == [(3, "simp", False)]


class TestExtractLineTimes:
    def test_extracts_time_for_matching_line(self):
        traces = [
            (0, "Elab.definition.value", 10.0, "my_thm"),
            (1, "Elab.step", 8.0, "simp [a, b]"),
        ]
        proof_items = [(2, "simp [a, b]", False)]
        line_times, total = _extract_line_times(traces, "my_thm", proof_items)
        assert total == 10.0
        assert line_times == {2: 8.0}

    def test_matches_bullets(self):
        traces = [
            (0, "Elab.definition.value", 10.0, "my_thm"),
            (1, "Elab.step", 5.0, "·"),  # bullet trace (empty after strip)
        ]
        proof_items = [(3, "ring", True)]  # is_bullet=True
        line_times, _ = _extract_line_times(traces, "my_thm", proof_items)
        assert line_times == {3: 5.0}

    def test_filters_expected_type(self):
        traces = [
            (0, "Elab.definition.value", 10.0, "my_thm"),
            (1, "Elab.step", 5.0, "expected type: Prop"),
            (1, "Elab.step", 8.0, "ring"),
        ]
        proof_items = [(2, "ring", False)]
        line_times, _ = _extract_line_times(traces, "my_thm", proof_items)
        assert line_times == {2: 8.0}

    def test_no_match_returns_empty(self):
        traces = [(0, "Elab.definition.value", 10.0, "other")]
        line_times, total = _extract_line_times(traces, "missing", [])
        assert line_times == {}
        assert total == 0.0

    @pytest.mark.parametrize(
        "name, traced",
        [
            ("my_thm'", "Foo.my_thm'"),
            ("my_thm!", "my_thm!"),
            ("my_thm?", "my_thm?"),
            ("x₁", "Foo.x₁"),
        ],
    )
    def test_matches_names_ending_in_any_name_character(self, name, traced):
        """`\\b` finds no word boundary after `'`, `!` or `?`, so such a
        theorem's value trace was never found and no line was reported."""
        traces = [
            (0, "Elab.definition.value", 10.0, f"{OK} {traced}"),
            (1, "Elab.step", 8.0, f"{OK} simp"),
        ]
        line_times, total = _extract_line_times(traces, name, [(2, "simp", False)])
        assert (line_times, total) == ({2: 8.0}, 10.0)

    @pytest.mark.parametrize(
        "traced", ["my_thm'", "my_thm!", "my_thm?", "my_thm₁", "Foo.my_thm'"]
    )
    def test_does_not_match_a_longer_name(self, traced):
        traces = [
            (0, "Elab.definition.value", 10.0, f"{OK} {traced}"),
            (1, "Elab.step", 8.0, f"{OK} simp"),
        ]
        assert _extract_line_times(traces, "my_thm", [(2, "simp", False)]) == (
            {},
            0.0,
        )

    def test_async_total_of_a_namespaced_theorem(self):
        """The trace names the theorem in full: `proof of Foo.my_thm'`."""
        traces = [(0, "Elab.async", 12.0, f"{OK} elaborating proof of Foo.my_thm'")]
        _, total = _extract_line_times(traces, "my_thm'", [])
        assert total == 12.0


class TestTraceStatusMarkers:
    """Messages arrive prefixed with the traced action's status emoji.

    The cases below use the marked form the profiler actually produces. The
    unmarked form used elsewhere in this file matches whatever the parser
    happens to do, which is why it never caught the prefix.
    """

    @pytest.mark.parametrize("marker", [OK, FAIL, ERROR], ids=["ok", "fail", "error"])
    def test_binds_a_tactic_whatever_its_status(self, marker):
        """With the marker left in place no message can match any source line,
        and a step that failed or threw still spent its time on that line."""
        traces = [
            (0, "Elab.definition.value", 10.0, "my_thm"),
            (1, "Elab.step", 8.0, f"{marker} induction n with"),
        ]
        proof_items = [(2, "induction n with", False)]

        line_times, _ = _extract_line_times(traces, "my_thm", proof_items)

        assert line_times == {2: 8.0}

    def test_keeps_a_leading_parenthesis(self):
        """Tactic text can start with punctuation, so only the status marker
        may be removed."""
        traces = [
            (0, "Elab.definition.value", 10.0, "my_thm"),
            (1, "Elab.step", 4.0, f"{OK} (rewrite [h])"),
        ]
        proof_items = [(2, "(rewrite [h])", False)]

        line_times, _ = _extract_line_times(traces, "my_thm", proof_items)

        assert line_times == {2: 4.0}

    def test_a_status_only_step_claims_nothing(self):
        """Lean wraps a tactic block in steps whose message is only the status
        prefix, and they arrive before any real tactic.

        Such a step must not take a line: an empty message matches a bullet on
        position alone, and matches any line at all through
        `content.startswith("")`. Skipping it is also what lets the first real
        tactic fix the depth the others are matched at.
        """
        traces = [
            (0, "Elab.definition.value", 10.0, "my_thm"),
            (1, "Elab.step", 9.0, OK),
            (2, "Elab.step", 8.0, f"{OK} norm_num"),
        ]
        proof_items = [(2, "exact hp", True), (3, "norm_num", True)]

        line_times, _ = _extract_line_times(traces, "my_thm", proof_items)

        assert line_times == {3: 8.0}

    def test_marked_expected_type_does_not_claim_the_tactic_depth(self):
        """The `expected type:` guard has to see the message without its marker,
        otherwise a pseudo step fixes the depth that real tactics are matched at."""
        traces = [
            (0, "Elab.definition.value", 10.0, "my_thm"),
            (1, "Elab.step", 5.0, f"{OK} expected type: Prop, term"),
            (2, "Elab.step", 8.0, f"{OK} ring"),
        ]
        proof_items = [(2, "ring", False)]

        line_times, _ = _extract_line_times(traces, "my_thm", proof_items)

        assert line_times == {2: 8.0}


class TestExtractTheoremSource:
    def test_extracts_single_theorem_with_imports(self):
        lines = [
            "import Mathlib",
            "open Nat",
            "",
            "theorem foo : True := by trivial",
            "",
            "theorem bar : True := by trivial",
        ]
        source, name, theorem_start = _extract_theorem_source(lines, 4)
        assert name == "foo"
        assert "import Mathlib" in source and "open Nat" in source
        assert "theorem foo" in source and "theorem bar" not in source
        # theorem is at line 4 in original, source has: 1:import 2:open 3:blank 4:blank 5:theorem
        assert theorem_start == 5

    def test_raises_on_non_theorem(self):
        with pytest.raises(ValueError, match="No theorem"):
            _extract_theorem_source(["-- comment"], 1)

    def test_theorem_start_without_header(self):
        """With no header lines the theorem sat one line lower than reported,
        which shifted every reported line by one."""
        lines = ["theorem foo : True := by", "  trivial"]
        source, _, theorem_start = _extract_theorem_source(lines, 1)
        assert source.splitlines()[theorem_start - 1] == lines[0]

    @pytest.mark.parametrize(
        "decl, name",
        [
            ("@[simp] theorem foo : True := by", "foo"),
            ("@[simp, aesop safe [constructors]] lemma foo : True := by", "foo"),
            ("protected theorem foo : True := by", "foo"),
            ("private nonrec def foo : True := by", "foo"),
            ("noncomputable def foo : Nat := by", "foo"),
            ("public theorem foo : True := by", "foo"),
            ("theorem foo.{u} (α : Sort u) : True := by", "foo"),
            ("theorem foo: True := by", "foo"),
            ("theorem _root_.foo : True := by", "foo"),
            ("theorem Nat.foo (n : Nat) : True := by", "Nat.foo"),
        ],
    )
    def test_reads_attributes_modifiers_and_name(self, decl, name):
        _, got, _ = _extract_theorem_source([decl, "  trivial"], 1)
        assert got == name

    @pytest.mark.parametrize(
        "next_command",
        [
            ["/-- Doc. -/", "theorem bar : True := by"],
            ["@[simp]", "theorem bar : True := by"],
            ["protected", "theorem bar : True := by"],
            ["set_option maxHeartbeats 400 in", "theorem bar : True := by"],
            ["open Nat in", "theorem bar : True := by"],
            ["unseal bar in", "theorem baz : True := by"],
            ["instance : Inhabited Nat := ⟨helper⟩"],
            ["grind_pattern foo => True"],
            ["example : True := trivial"],
            ["end B"],
            ["namespace C"],
            ["noncomputable section"],
            ["variable (n : Nat)"],
            ["#check foo"],
            ["/-! ## Next part -/"],
        ],
    )
    def test_ends_at_the_next_command(self, next_command):
        """Cutting only at the next theorem/lemma/def left its doc comment or
        attribute dangling, or kept an `end`, `instance`, ... that cannot
        elaborate in the snippet."""
        lines = [*HEADER, "theorem foo : True := by", "  trivial", "", *next_command]
        assert _theorem_lines(lines, 3) == lines[2:5]

    def test_keeps_tactics_that_look_like_commands(self):
        lines = [
            *HEADER,
            "theorem foo : True := by",
            "  open Classical in",
            "  set_option maxRecDepth 100 in",
            "  trivial",
        ]
        assert _theorem_lines(lines, 3) == lines[2:]

    def test_ends_at_the_next_command_at_the_same_indentation(self):
        lines = [
            "namespace Foo",
            "  theorem foo : True := by",
            "    trivial",
            "  @[simp] theorem bar : True := by",
            "    trivial",
            "end Foo",
        ]
        assert _theorem_lines(lines, 2) == lines[1:3]

    def test_skips_commented_out_code(self):
        """A theorem inside a block comment cut the snippet inside the comment."""
        lines = [
            *HEADER,
            "theorem foo : True := by",
            "  trivial",
            "",
            "/-",
            "theorem old : True := by",
            "  trivial",
            "-/",
        ]
        assert _theorem_lines(lines, 3) == lines[2:]

    def test_header_continues_after_a_section_with_attributes(self):
        """The header stopped at `@[expose] public section`, which dropped the
        `namespace` and `open` lines after it."""
        lines = [
            "module",
            "",
            "public import Lean",
            "",
            "@[expose] public section",
            "",
            "namespace Foo",
            "",
            "open Nat",
            "",
            "theorem foo : True := by",
            "  trivial",
        ]
        source, _, theorem_start = _extract_theorem_source(lines, 11)
        assert source.splitlines()[: theorem_start - 1] == lines[:10] + [""]

    def test_header_keeps_a_structure_with_protected_fields(self):
        """A `protected` field ended the header in the middle of the class."""
        lines = [
            *HEADER,
            "class Foo (α : Type) where",
            "  foo : α",
            "  protected foo_eq : foo = foo",
            "",
            "theorem bar : True := by",
            "  trivial",
        ]
        source, _, theorem_start = _extract_theorem_source(lines, 7)
        assert source.splitlines()[: theorem_start - 1] == lines[:6] + [""]

    def test_header_keeps_a_structure_with_documented_fields(self):
        """A doc comment followed by code on its line is not the start of the
        next declaration."""
        lines = [
            *HEADER,
            "structure Foo where",
            "  /-- The constructor. -/ mk ::",
            "  /-- The value. -/ val : Nat",
            "",
            "/-- Doc. -/",
            "theorem foo : True := by",
            "  trivial",
        ]
        source, _, _ = _extract_theorem_source(lines, 8)
        assert source.splitlines() == [*lines[:6], "", *lines[6:]]

    def test_block_comments_nest(self):
        """A module doc that shows an example with a doc comment, `namespace`
        and `end` is one comment; it ended at the example's `-/`."""
        lines = [
            *HEADER,
            "/-!",
            "```",
            "namespace MyHom",
            "/-- Doc. -/",
            "protected def copy := 1",
            "end MyHom",
            "```",
            "-/",
            "",
            "namespace Foo",
            "",
            "theorem foo : True := by",
            "  trivial",
        ]
        source, _, theorem_start = _extract_theorem_source(lines, 14)
        assert source.splitlines()[: theorem_start - 1] == lines[:13] + [""]

    def test_line_comments_open_no_block_comment(self):
        lines = [
            *HEADER,
            "theorem foo : True := by",
            "  -- r = +/- j ^ 2",
            "  trivial",
            "",
            "theorem bar : True := by",
            "  trivial",
        ]
        assert _theorem_lines(lines, 3) == lines[2:6]

    def test_dash_dash_slash_closes_a_block_comment(self):
        """Inside a block comment `--` starts no line comment, so `--/` closes
        the comment; the rest of the file was taken as commented out."""
        assert profile_utils._comment_depth("note --/", 1) == 0
        lines = [
            *HEADER,
            "theorem foo : True := by",
            "  trivial",
            "",
            "/- note --/",
            "",
            "theorem bar : True := by",
            "  trivial",
        ]
        assert _theorem_lines(lines, 3) == lines[2:7]

    def test_a_trailing_comment_is_not_an_in_prefix(self):
        """`namespace Foo -- results used in Bar` was dropped as an `... in`
        command while its `end Foo` was replayed, which Lean rejects."""
        lines = [
            *HEADER,
            "theorem a : True := by",
            "  trivial",
            "",
            "namespace Foo -- results used in Bar",
            "theorem b : True := by",
            "  trivial",
            "end Foo",
            "",
            "theorem foo : True := by",
            "  trivial",
        ]
        source, _, _ = _extract_theorem_source(lines, 11)
        assert source.splitlines() == [*HEADER, "", lines[5], lines[8], *lines[10:]]

    def test_keeps_lines_above_the_theorem_that_end_in_a_comment(self):
        lines = [
            *HEADER,
            "theorem a : True := by",
            "  trivial",
            "",
            "set_option maxRecDepth 2000 in -- needed in the proof",
            "open Nat in -- for succ_pos",
            "@[simp] -- a simp lemma",
            "theorem foo : True := by",
            "  trivial",
        ]
        source, _, theorem_start = _extract_theorem_source(lines, 9)
        assert source.splitlines()[:theorem_start][-4:] == lines[5:9]

    def test_replays_a_multi_line_variable_with_a_comment_and_a_gap(self):
        lines = [
            *HEADER,
            "theorem a : True := by",
            "  trivial",
            "",
            "variable {α : Type}",
            "  -- the instance",
            "  [Inhabited α]",
            "",
            "  (x : α)",
            "theorem foo : (default : α) = default := by",
            "  rfl",
        ]
        source, _, _ = _extract_theorem_source(lines, 11)
        assert source.splitlines() == [
            *HEADER,
            "",
            lines[5],
            lines[7],
            lines[9],
            *lines[10:],
        ]

    def test_replays_scope_commands_before_the_theorem(self):
        """Namespace, open and variable commands after the first declaration
        decide what the theorem's names resolve to. Declarations, `... in`
        prefixes of other commands and the first declaration's doc comment are
        left out."""
        lines = [
            *HEADER,
            "namespace A",
            "/-- Doc of a. -/",
            "theorem a : True := by",
            "  open Classical in",
            "  trivial",
            "end A",
            "",
            "namespace B",
            "open Nat in",
            "theorem b : True := by",
            "  trivial",
            "variable {α : Type}",
            "  [Inhabited α]",
            "theorem foo : (default : α) = default := by",
            "  rfl",
            "end B",
        ]
        source, _, theorem_start = _extract_theorem_source(lines, 16)
        assert source.splitlines() == [
            *HEADER,
            "namespace A",
            "",
            "end A",
            "namespace B",
            "variable {α : Type}",
            "  [Inhabited α]",
            "theorem foo : (default : α) = default := by",
            "  rfl",
        ]
        assert theorem_start == 9

    def test_skips_mutual_blocks(self):
        """The `end` of a `mutual` block closes no namespace or section."""
        lines = [
            "mutual",
            "def f : Nat → Nat",
            "  | 0 => 0",
            "  | n + 1 => g n",
            "def g : Nat → Nat",
            "  | 0 => 0",
            "  | n + 1 => f n",
            "end",
            "",
            "theorem foo : True := by",
            "  trivial",
        ]
        source, _, _ = _extract_theorem_source(lines, 10)
        assert source.splitlines() == ["", *lines[9:]]

    def test_keeps_the_lines_written_above_the_theorem(self):
        """`include h in`, `open ... in` and `set_option ... in` above the
        theorem apply to it, as do its doc comment, attributes and modifiers."""
        lines = [
            *HEADER,
            "theorem a : True := by",
            "  trivial",
            "",
            "set_option maxHeartbeats 400000 in",
            "include h in",
            "/-- Doc. -/",
            "@[simp]",
            "private",
            "theorem foo : True := by",
            "  exact h",
        ]
        source, _, theorem_start = _extract_theorem_source(lines, 11)
        assert source.splitlines()[:theorem_start][-6:] == lines[5:11]
        assert source.splitlines()[theorem_start:] == lines[11:]


class TestProfileTheorem:
    """profile_theorem on canned `lean --profile` output."""

    @pytest.fixture
    def lean_output(self, monkeypatch) -> list[str]:
        output: list[str] = []

        async def fake_run(file_path, project_path, timeout):
            return "".join(output)

        monkeypatch.setattr(profile_utils, "_run_lean_profile", fake_run)
        return output

    async def test_reports_original_line_numbers_without_header(
        self, tmp_path: Path, lean_output: list[str]
    ):
        lean_file = tmp_path / "T.lean"
        lean_file.write_text("theorem foo : True := by\n  trivial\n", encoding="utf-8")
        lean_output.append(
            f"[Elab.definition.value] [0.010] {OK} foo\n"
            f"  [Elab.step] [0.008] {OK} trivial\n"
        )

        profile = await profile_theorem(lean_file, 1, tmp_path)

        assert [(ln.line, ln.text) for ln in profile.lines] == [(2, "trivial")]

    async def test_proof_lines_come_from_the_theorem_not_its_doc(
        self, tmp_path: Path, lean_output: list[str]
    ):
        """The proof was looked for from the top of the snippet, so a doc
        comment line ending in " by" was taken as the start of the proof."""
        lean_file = tmp_path / "T.lean"
        lean_file.write_text(
            "/-- Zero is a right identity, proved by\n"
            "simp. -/\n"
            "theorem foo (n : Nat) : n + 0 = n := by\n"
            "  simp\n",
            encoding="utf-8",
        )
        lean_output.append(
            f"[Elab.definition.value] [0.010] {OK} foo\n"
            f"  [Elab.step] [0.008] {OK} simp\n"
        )

        profile = await profile_theorem(lean_file, 3, tmp_path)

        assert [(ln.line, ln.text) for ln in profile.lines] == [(4, "simp")]

    async def test_parse_error_does_not_blame_same_file_dependencies(
        self, tmp_path: Path, lean_output: list[str]
    ):
        lean_file = tmp_path / "T.lean"
        lean_file.write_text("theorem foo : True := by\n  trivial\n", encoding="utf-8")
        lean_output.append("T.lean:4:0: error: unexpected end of input\n")

        with pytest.raises(ValueError, match="failed to parse") as excinfo:
            await profile_theorem(lean_file, 1, tmp_path)

        assert "same-file" not in str(excinfo.value)

    async def test_elaboration_error_names_same_file_dependencies(
        self, tmp_path: Path, lean_output: list[str]
    ):
        lean_file = tmp_path / "T.lean"
        lean_file.write_text("theorem foo : True := by\n  exact h\n", encoding="utf-8")
        lean_output.append(
            "T.lean:3:8: error(lean.unknownIdentifier): Unknown identifier `h`\n"
        )

        with pytest.raises(ValueError, match="same-file dependencies"):
            await profile_theorem(lean_file, 1, tmp_path)


class TestFilterCategories:
    def test_filters_skip_and_small(self):
        cumulative = {"import": 100.0, "simp": 5.0, "parsing": 2.0, "ring": 0.5}
        assert _filter_categories(cumulative) == {"simp": 5.0}


class _FakeProc:
    """Stands in for the `lake env lean --profile` subprocess."""

    pid = 4321

    def __init__(self):
        self.killed = False

    async def communicate(self):
        return b"[Elab.command] [0.005] done\n", None

    def kill(self):
        self.killed = True


class TestRunLeanProfileCleanup:
    """The cleanup in _run_lean_profile's `finally` runs on every profile.

    Both platforms are simulated rather than detected, so each case is
    exercised wherever the suite runs.
    """

    @pytest.fixture
    def proc(self, monkeypatch) -> _FakeProc:
        fake = _FakeProc()

        async def fake_exec(*args, **kwargs):
            return fake

        monkeypatch.setattr(profile_utils.asyncio, "create_subprocess_exec", fake_exec)
        return fake

    async def test_windows_kills_the_process_instead_of_the_group(
        self, monkeypatch, tmp_path: Path, proc: _FakeProc
    ):
        """Windows has no os.killpg, so the group kill must not be attempted.

        Calling it raised AttributeError out of the `finally`, which discarded
        the profile that had already been produced.
        """
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.delattr(os, "killpg", raising=False)
        monkeypatch.delattr(os, "getpgid", raising=False)

        output = await _run_lean_profile(tmp_path / "T.lean", tmp_path, 30.0)

        assert output == "[Elab.command] [0.005] done\n"
        assert proc.killed

    async def test_posix_still_kills_the_whole_group(
        self, monkeypatch, tmp_path: Path, proc: _FakeProc
    ):
        """The group kill is what stops lean surviving as an orphan."""
        killed: list[tuple[int, int]] = []
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(os, "getpgid", lambda pid: pid, raising=False)
        monkeypatch.setattr(
            os, "killpg", lambda pgid, sig: killed.append((pgid, sig)), raising=False
        )

        output = await _run_lean_profile(tmp_path / "T.lean", tmp_path, 30.0)

        assert output == "[Elab.command] [0.005] done\n"
        assert killed == [(_FakeProc.pid, 9)]
        assert not proc.killed

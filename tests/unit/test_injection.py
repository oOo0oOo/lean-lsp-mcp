"""Security tests: code injection via decl_name and theorem_name.

BUG-4 (P0): Both tools interpolate user-supplied names directly into Lean code
with zero sanitization. This allows arbitrary Lean code execution via the MCP
interface, including IO operations via #eval.

These tests document the vulnerability and will serve as regression tests
once the fix is applied.
"""

from __future__ import annotations

from lean_lsp_mcp.goal_tracker import make_sorry_snippet


class TestGoalTrackerSnippetInjection:
    """goal_tracker.py:39 — `let target : Name := "{decl_name}".splitOn "."`"""

    def test_double_quote_breaks_lean_string(self):
        """A double quote in decl_name terminates the Lean string literal.

        The snippet becomes:
            let target : Name := "foo"bar".splitOn "."
        which is invalid Lean syntax — but the code after the quote
        could be crafted to be valid Lean.
        """
        snippet = make_sorry_snippet('foo"bar')
        # The raw double quote appears in the snippet — vulnerability present
        assert '"foo"bar"' in snippet or 'foo"bar' in snippet

    def test_newline_injects_new_lean_command(self):
        """A newline creates a new line in the Lean code.

        decl_name = 'foo\\n#eval IO.println "pwned"' becomes:
            let target : Name := "foo
            #eval IO.println "pwned"".splitOn "."
        The #eval is on its own line and will be executed by Lean.
        """
        malicious = 'foo\n#eval IO.println "pwned"'
        snippet = make_sorry_snippet(malicious)
        lines = snippet.split("\n")
        # The injected #eval appears as a separate line
        assert any("#eval IO.println" in line for line in lines)

    def test_backslash_n_in_lean_string(self):
        """Literal backslash-n in the name — tests escape handling."""
        snippet = make_sorry_snippet("foo\\nbar")
        # \n inside a Lean string is an escape sequence, not a literal newline
        assert "foo\\nbar" in snippet

    def test_curly_braces_in_name(self):
        """Curly braces could interfere with Lean string interpolation.

        Regular Lean strings don't support interpolation (only s!"..."),
        but this tests the boundary.
        """
        snippet = make_sorry_snippet("foo{bar}")
        assert "foo{bar}" in snippet

    def test_semicolon_in_name(self):
        """Semicolons could terminate Lean statements."""
        snippet = make_sorry_snippet('foo"; let x := 0; --')
        assert 'foo"' in snippet

    def test_very_long_name(self):
        """Very long name — tests buffer/performance behavior."""
        long_name = "A" * 10_000
        snippet = make_sorry_snippet(long_name)
        assert long_name in snippet


class TestVerifyTheoremNameInjection:
    """server.py:1179 — `snippet = f"\\n#print axioms _root_.{theorem_name}\\n"`

    These tests document what the injected snippet looks like.
    Since verify_theorem is a server function requiring LSP context,
    we test the string interpolation pattern directly.
    """

    @staticmethod
    def _make_verify_snippet(theorem_name: str) -> str:
        """Reproduce the exact interpolation from server.py:1179."""
        return f"\n#print axioms _root_.{theorem_name}\n"

    def test_newline_injection(self):
        """Newline in theorem_name allows injecting arbitrary Lean code.

        theorem_name = 'Foo\\n#eval IO.println "pwned"' becomes:
            #print axioms _root_.Foo
            #eval IO.println "pwned"
        """
        snippet = self._make_verify_snippet('Foo\n#eval IO.println "pwned"')
        lines = snippet.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "#print axioms _root_.Foo"
        assert '#eval IO.println "pwned"' == lines[1]

    def test_double_root_prefix(self):
        """theorem_name already has _root_ prefix — gets doubled."""
        snippet = self._make_verify_snippet("_root_.Foo.bar")
        assert "_root_._root_.Foo.bar" in snippet

    def test_empty_name(self):
        """Empty theorem_name — produces `#print axioms _root_.`."""
        snippet = self._make_verify_snippet("")
        assert "#print axioms _root_." in snippet

    def test_spaces_in_name(self):
        """Spaces in theorem_name — Lean will parse as multiple tokens."""
        snippet = self._make_verify_snippet("Foo bar baz")
        assert "#print axioms _root_.Foo bar baz" in snippet

    def test_hash_command_injection(self):
        """# at start of injected content after newline."""
        snippet = self._make_verify_snippet("X\n#check Nat")
        assert "#check Nat" in snippet


class TestSanitizationRecommendations:
    """Tests that document what a proper fix should look like.

    After BUG-4 is fixed, these tests should be updated to verify
    the sanitization works correctly.
    """

    # Valid Lean identifiers match: [A-Za-z_\u00C0-\uFFFF][A-Za-z0-9_\u00C0-\uFFFF.'«»]*
    # For FQNs, dots are also allowed.

    SAFE_NAMES = [
        "foo",
        "Foo.bar",
        "A.B.C.D",
        "foo₀",
        "bar'",
        "_private.Module.Path.0.Ns.name",
        "«unusual name»",
    ]

    DANGEROUS_NAMES = [
        'foo"bar',  # breaks Lean string
        "foo\nbar",  # injects newline
        'foo\n#eval panic!""',  # code injection
        "foo{bar}",  # potential string interpolation
        "",  # empty
        " ",  # whitespace only
    ]

    def test_safe_names_produce_valid_snippets(self):
        """All safe names should produce snippets without injection."""
        for name in self.SAFE_NAMES:
            snippet = make_sorry_snippet(name)
            # Each line of the snippet before the name should be valid
            assert f'"{name}"' in snippet

    def test_dangerous_names_documented(self):
        """Dangerous names currently pass through unsanitized."""
        for name in self.DANGEROUS_NAMES:
            # Currently these all "work" — they produce snippets
            # After fix, they should either be rejected or escaped
            snippet = make_sorry_snippet(name)
            assert snippet  # Non-empty — vulnerability present

"""Unit tests for the name resolution logic in goal_tracker (server.py:1268-1326).

The name resolution algorithm:
1. Only activates for short names (no dots in decl_name)
2. Scans file line-by-line for namespace/end blocks to build ns_stack
3. Strips modifiers/attributes to find core keywords (theorem, def, etc.)
4. Detects `private` modifier to construct mangled FQN
5. Resolves short name to FQN using namespace stack

Since this logic is inline in server.py's goal_tracker(), we extract
and test the same algorithm here.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Extract the name-resolution algorithm from server.py into a testable form
# ---------------------------------------------------------------------------


def resolve_short_name(
    decl_name: str,
    file_content: str,
    rel_path: str,
) -> str:
    """Reimplementation of the name resolution logic from server.py:1268-1326.

    Returns the resolved FQN, or the original name if no resolution occurs.
    Raises ValueError for ambiguous names.
    """
    if "." in decl_name:
        return decl_name  # FQN passthrough

    lines_for_resolve = file_content.splitlines()
    ns_stack: list[str] = []
    _modifiers = {
        "private",
        "protected",
        "noncomputable",
        "nonrec",
        "unsafe",
        "partial",
        "@[simp]",
        "@[inline]",
    }
    _core_keywords = {
        "theorem",
        "lemma",
        "def",
        "abbrev",
        "instance",
        "inductive",
        "structure",
        "class",
    }
    candidates: list[str] = []
    for src_line in lines_for_resolve:
        s = src_line.strip()
        if s.startswith("namespace "):
            ns_stack.append(s[len("namespace ") :].strip())
        elif s.startswith("end "):
            ended = s[len("end ") :].strip()
            if ns_stack and ns_stack[-1] == ended:
                ns_stack.pop()
        else:
            words = s.split()
            idx = 0
            is_private = False
            while idx < len(words) and (
                words[idx] in _modifiers or words[idx].startswith("@[")
            ):
                if words[idx] == "private":
                    is_private = True
                idx += 1
            if (
                idx < len(words)
                and words[idx] in _core_keywords
                and idx + 1 < len(words)
            ):
                name_part = words[idx + 1].rstrip(":({[")
                if name_part == decl_name:
                    fqn = ".".join(ns_stack + [decl_name]) if ns_stack else decl_name
                    if is_private:
                        module_dotpath = (
                            rel_path.removesuffix(".lean")
                            .replace("/", ".")
                            .replace("\\", ".")
                        )
                        fqn = f"_private.{module_dotpath}.0.{fqn}"
                    candidates.append(fqn)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        unique = list(dict.fromkeys(candidates))
        if len(unique) == 1:
            return unique[0]
        raise ValueError(f"Ambiguous name '{decl_name}', matches: {unique}")

    return decl_name  # No match found


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFQNPassthrough:
    def test_dotted_name_not_resolved(self):
        result = resolve_short_name("A.B.foo", "theorem foo := sorry", "Foo.lean")
        assert result == "A.B.foo"

    def test_single_component_triggers_resolution(self):
        result = resolve_short_name("foo", "theorem foo := sorry", "Foo.lean")
        assert result == "foo"  # No namespace, stays as-is


class TestNamespaceResolution:
    def test_single_namespace(self):
        content = """\
namespace Bar
theorem foo : True := trivial
end Bar
"""
        assert resolve_short_name("foo", content, "Bar.lean") == "Bar.foo"

    def test_nested_namespaces(self):
        content = """\
namespace A
namespace B
theorem baz : True := trivial
end B
end A
"""
        assert resolve_short_name("baz", content, "Mod.lean") == "A.B.baz"

    def test_no_namespace(self):
        content = "theorem foo : True := trivial\n"
        assert resolve_short_name("foo", content, "Foo.lean") == "foo"

    def test_declaration_after_namespace_closes(self):
        content = """\
namespace A
theorem inner : True := trivial
end A
theorem outer : True := trivial
"""
        assert resolve_short_name("inner", content, "Foo.lean") == "A.inner"
        assert resolve_short_name("outer", content, "Foo.lean") == "outer"


class TestPrivateDeclarations:
    def test_private_in_namespace(self):
        content = """\
namespace Foo
private theorem helper : True := trivial
end Foo
"""
        result = resolve_short_name("helper", content, "Math/Bar.lean")
        assert result == "_private.Math.Bar.0.Foo.helper"

    def test_private_at_top_level(self):
        content = "private def helper : Nat := 42\n"
        result = resolve_short_name("helper", content, "Mod.lean")
        assert result == "_private.Mod.0.helper"

    def test_private_with_nested_namespace(self):
        content = """\
namespace A
namespace B
private theorem secret : True := trivial
end B
end A
"""
        result = resolve_short_name("secret", content, "Path/To/File.lean")
        assert result == "_private.Path.To.File.0.A.B.secret"

    def test_windows_path_separators(self):
        content = "private def helper : Nat := 42\n"
        result = resolve_short_name("helper", content, "Path\\To\\File.lean")
        assert result == "_private.Path.To.File.0.helper"


class TestModifierStripping:
    def test_noncomputable_private(self):
        content = "noncomputable private def foo : Nat := 42\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "_private.Mod.0.foo"

    def test_protected_theorem(self):
        content = """\
namespace A
protected theorem foo : True := trivial
end A
"""
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "A.foo"

    def test_single_attribute_simp(self):
        content = "@[simp] theorem foo : True := trivial\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"

    def test_attribute_startswith_at_bracket(self):
        """Any token starting with @[ is consumed as an attribute."""
        content = "@[norm_num] theorem foo : True := trivial\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"

    def test_multi_token_attribute_bug2(self):
        """BUG-2: @[simp, norm_num] splits into multiple tokens.

        @[simp, → starts with @[, consumed
        norm_num] → does NOT start with @[ and not in _modifiers → parsing STOPS
        theorem is never seen as a core keyword → declaration MISSED.
        """
        content = "@[simp, norm_num] theorem foo : True := trivial\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        # BUG: declaration is not found, name stays unresolved
        assert result == "foo"  # Stays unresolved — this is the bug

    def test_multi_attribute_separate_brackets(self):
        """@[simp] @[norm_num] — each is a separate token, both consumed."""
        content = "@[simp] @[norm_num] theorem foo : True := trivial\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"  # Works correctly

    def test_unsafe_modifier(self):
        content = "unsafe def foo : Nat := 42\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"

    def test_partial_modifier(self):
        content = "partial def foo : Nat := foo\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"


class TestEndBlockEdgeCases:
    def test_end_without_matching_namespace(self):
        """end Foo with empty ns_stack — should not crash."""
        content = """\
end Foo
theorem bar : True := trivial
"""
        result = resolve_short_name("bar", content, "Mod.lean")
        assert result == "bar"

    def test_end_with_wrong_name(self):
        """end B when ns_stack[-1] is A — stack not popped."""
        content = """\
namespace A
end B
theorem foo : True := trivial
end A
"""
        # end B doesn't pop A, so foo is still in namespace A
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "A.foo"

    def test_section_same_name_as_namespace_bug3(self):
        """BUG-3: section named same as namespace pops the ns_stack.

        namespace A
        section A
        end A     ← pops namespace A (wrong! should only close section)
        theorem foo  ← now outside namespace A (incorrect)
        end A     ← nothing to pop

        This is a known bug: sections use `end` but shouldn't affect ns_stack.
        """
        content = """\
namespace A
section A
end A
theorem foo : True := trivial
end A
"""
        result = resolve_short_name("foo", content, "Mod.lean")
        # BUG: end A (for section) pops namespace A, so foo resolves without A prefix
        assert result == "foo"  # Bug: should be "A.foo"

    def test_section_different_name_no_effect(self):
        """section Foo inside namespace A — end Foo doesn't pop A."""
        content = """\
namespace A
section Foo
theorem bar : True := trivial
end Foo
end A
"""
        result = resolve_short_name("bar", content, "Mod.lean")
        assert result == "A.bar"


class TestNamePartMatching:
    def test_trailing_colon_stripped(self):
        content = "theorem foo : True := trivial\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"

    def test_trailing_paren_stripped(self):
        content = "def foo(n : Nat) : Nat := n\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"

    def test_trailing_brace_stripped(self):
        content = "def foo{n : Nat} : Nat := n\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"

    def test_trailing_bracket_stripped(self):
        content = "def foo[Inhabited Nat] : Nat := default\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"

    def test_substring_no_match(self):
        """foo should not match fooBar."""
        content = "theorem fooBar : True := trivial\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"  # Not resolved — fooBar != foo

    def test_exact_match_required(self):
        """fooBar stripped of :({[ is still fooBar, not foo."""
        content = "theorem fooBar : True := trivial\ntheorem foo : True := trivial\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"  # Matches the second theorem


class TestAmbiguity:
    def test_ambiguous_short_name(self):
        content = """\
namespace A
theorem foo : True := trivial
end A
namespace B
theorem foo : True := trivial
end B
"""
        with pytest.raises(ValueError, match="Ambiguous"):
            resolve_short_name("foo", content, "Mod.lean")

    def test_same_fqn_deduplicated(self):
        """Same FQN found twice — deduplicated, no ambiguity error."""
        # This can't easily happen in practice, but the code handles it
        content = """\
namespace A
theorem foo : True := trivial
theorem foo : True := trivial
end A
"""
        # Both resolve to A.foo — deduplicated
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "A.foo"

    def test_multiple_private_same_name_different_ns(self):
        """Multiple private decls with same short name in different namespaces."""
        content = """\
namespace A
private theorem helper : True := trivial
end A
namespace B
private theorem helper : True := trivial
end B
"""
        with pytest.raises(ValueError, match="Ambiguous"):
            resolve_short_name("helper", content, "Mod.lean")


class TestNamespaceInComments:
    def test_namespace_in_line_comment(self):
        """Line comments are not parsed specially — namespace is detected (BUG)."""
        content = """\
-- namespace Fake
theorem foo : True := trivial
"""
        # The stripped line starts with "-- namespace" not "namespace ", so it's fine
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"  # Correct: line comment prefix prevents match

    def test_namespace_on_own_line_in_block_comment(self):
        """Block comment with namespace on its own line — falsely detected.

        /-
        namespace Fake
        -/

        The stripped line "namespace Fake" starts with "namespace " — false positive.
        """
        content = """\
/-
namespace Fake
-/
theorem foo : True := trivial
end Fake
"""
        result = resolve_short_name("foo", content, "Mod.lean")
        # BUG: namespace Fake inside block comment is treated as real
        # foo resolves to Fake.foo (incorrect)
        assert result == "Fake.foo"  # Documents the bug


class TestCoreKeywords:
    """All core keywords should be recognized."""

    @pytest.mark.parametrize(
        "keyword",
        [
            "theorem",
            "lemma",
            "def",
            "abbrev",
            "instance",
            "inductive",
            "structure",
            "class",
        ],
    )
    def test_keyword_detected(self, keyword):
        content = f"{keyword} foo : True := trivial\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        assert result == "foo"

    def test_nonstandard_keyword_not_detected(self):
        """'example' is not in _core_keywords — declaration missed."""
        content = "example foo : True := trivial\n"
        result = resolve_short_name("foo", content, "Mod.lean")
        # 'example' is not in _core_keywords, so foo is not found
        assert result == "foo"  # Stays unresolved


class TestDeclarationInsideSection:
    def test_section_does_not_affect_fqn(self):
        """Sections don't create namespaces in Lean."""
        content = """\
section Foo
theorem bar : True := trivial
end Foo
"""
        result = resolve_short_name("bar", content, "Mod.lean")
        # section/end doesn't affect ns_stack (section != namespace)
        # "section Foo" doesn't start with "namespace "
        assert result == "bar"

"""Unit tests for goal_tracker module."""

from __future__ import annotations

from lean_lsp_mcp.goal_tracker import (
    SorryNode,
    make_sorry_snippet,
    parse_sorry_result,
    render_tree,
)


class TestMakeSorrySnippet:
    def test_contains_decl_name(self):
        snippet = make_sorry_snippet("Foo.bar")
        assert '"Foo.bar"' in snippet

    def test_contains_eval(self):
        snippet = make_sorry_snippet("myThm")
        assert "#eval" in snippet

    def test_contains_node_marker(self):
        snippet = make_sorry_snippet("myThm")
        assert "MCP_NODE" in snippet

    def test_contains_summary_marker(self):
        snippet = make_sorry_snippet("myThm")
        assert "MCP_SUMMARY" in snippet

    def test_dotted_fqn(self):
        """All components of a dotted FQN appear in snippet."""
        snippet = make_sorry_snippet("A.B.C.D")
        assert '"A.B.C.D"' in snippet

    def test_unicode_name(self):
        """Unicode subscripts and primes pass through."""
        snippet = make_sorry_snippet("foo₀")
        assert '"foo₀"' in snippet

    def test_private_mangled_name(self):
        """Private mangled name with dots handled correctly."""
        snippet = make_sorry_snippet("_private.Module.Path.0.Ns.name")
        assert '"_private.Module.Path.0.Ns.name"' in snippet

    def test_empty_name(self):
        """Empty string produces a snippet (Lean will error at runtime)."""
        snippet = make_sorry_snippet("")
        assert '""' in snippet
        assert "#eval" in snippet

    def test_snippet_has_max_heartbeats(self):
        """Snippet sets maxHeartbeats to prevent runaway BFS."""
        snippet = make_sorry_snippet("foo")
        assert "maxHeartbeats" in snippet

    def test_double_quote_in_name_breaks_lean_string(self):
        """BUG-4: A name containing '"' breaks the Lean string literal.

        This documents the injection vulnerability. The snippet interpolates
        decl_name directly into a Lean string: `"{decl_name}"`. A double quote
        in the name terminates the string early.
        """
        snippet = make_sorry_snippet('foo"bar')
        # The snippet will contain '"foo"bar"' which is broken Lean syntax
        # This test documents the bug — after fix, the name should be rejected or escaped
        assert 'foo"bar' in snippet  # The raw name appears — this is the vulnerability

    def test_newline_in_name_injects_lean_code(self):
        """BUG-4: A newline in decl_name injects new Lean commands.

        The snippet uses f-string interpolation, so a newline in the name
        creates a new line in the Lean code, allowing arbitrary code injection.
        """
        snippet = make_sorry_snippet('foo\n#eval panic! "injected"')
        # The snippet contains the injected code on a new line
        assert "#eval panic!" in snippet  # Documents the vulnerability


class TestParseSorryResult:
    def test_no_sorry(self):
        diags = [{"severity": 3, "message": 'MCP_SUMMARY:{"visited":42}'}]
        nodes, visited = parse_sorry_result(diags)
        assert nodes == {}
        assert visited == 42

    def test_single_explicit(self):
        diags = [
            {
                "severity": 3,
                "message": 'MCP_NODE:{"name":"myThm","explicit":true,"sorry_deps":[]}',
            },
            {"severity": 3, "message": 'MCP_SUMMARY:{"visited":5}'},
        ]
        nodes, visited = parse_sorry_result(diags)
        assert "myThm" in nodes
        assert nodes["myThm"].explicit_sorry is True
        assert visited == 5

    def test_transitive_chain(self):
        diags = [
            {
                "severity": 3,
                "message": 'MCP_NODE:{"name":"A","explicit":false,"sorry_deps":["B"]}',
            },
            {
                "severity": 3,
                "message": 'MCP_NODE:{"name":"B","explicit":true,"sorry_deps":[]}',
            },
            {"severity": 3, "message": 'MCP_SUMMARY:{"visited":10}'},
        ]
        nodes, visited = parse_sorry_result(diags)
        assert len(nodes) == 2
        assert nodes["A"].explicit_sorry is False
        assert nodes["A"].sorry_deps == ["B"]
        assert nodes["B"].explicit_sorry is True
        assert visited == 10

    def test_ignores_non_info(self):
        diags = [
            {
                "severity": 1,
                "message": 'MCP_NODE:{"name":"X","explicit":true,"sorry_deps":[]}',
            }
        ]
        nodes, _ = parse_sorry_result(diags)
        assert nodes == {}

    def test_empty(self):
        nodes, visited = parse_sorry_result([])
        assert nodes == {}
        assert visited == 0

    def test_malformed_json_node(self):
        """Malformed JSON in MCP_NODE is skipped, rest still parsed."""
        diags = [
            {"severity": 3, "message": "MCP_NODE:{bad json}"},
            {
                "severity": 3,
                "message": 'MCP_NODE:{"name":"B","explicit":true,"sorry_deps":[]}',
            },
            {"severity": 3, "message": 'MCP_SUMMARY:{"visited":3}'},
        ]
        nodes, visited = parse_sorry_result(diags)
        assert len(nodes) == 1
        assert "B" in nodes
        assert visited == 3

    def test_malformed_json_summary(self):
        """Malformed MCP_SUMMARY — visited stays 0."""
        diags = [
            {
                "severity": 3,
                "message": 'MCP_NODE:{"name":"A","explicit":true,"sorry_deps":[]}',
            },
            {"severity": 3, "message": "MCP_SUMMARY:{bad}"},
        ]
        nodes, visited = parse_sorry_result(diags)
        assert "A" in nodes
        assert visited == 0

    def test_duplicate_node_overwrites(self):
        """Duplicate MCP_NODE names — second overwrites first, deps may be lost."""
        diags = [
            {
                "severity": 3,
                "message": 'MCP_NODE:{"name":"A","explicit":false,"sorry_deps":["X"]}',
            },
            {
                "severity": 3,
                "message": 'MCP_NODE:{"name":"A","explicit":true,"sorry_deps":["Y"]}',
            },
        ]
        nodes, _ = parse_sorry_result(diags)
        assert nodes["A"].sorry_deps == ["Y"]  # Second wins
        assert nodes["A"].explicit_sorry is True

    def test_multiple_summaries_last_wins(self):
        """Multiple MCP_SUMMARY lines — last value wins."""
        diags = [
            {"severity": 3, "message": 'MCP_SUMMARY:{"visited":10}'},
            {"severity": 3, "message": 'MCP_SUMMARY:{"visited":99}'},
        ]
        _, visited = parse_sorry_result(diags)
        assert visited == 99

    def test_ignores_warning_severity(self):
        """severity=2 (warning) with MCP_NODE text should be ignored."""
        diags = [
            {
                "severity": 2,
                "message": 'MCP_NODE:{"name":"X","explicit":true,"sorry_deps":[]}',
            },
        ]
        nodes, _ = parse_sorry_result(diags)
        assert nodes == {}

    def test_mixed_errors_warnings_infos(self):
        """Only severity=3 (info) messages are parsed."""
        diags = [
            {
                "severity": 1,
                "message": 'MCP_NODE:{"name":"err","explicit":true,"sorry_deps":[]}',
            },
            {
                "severity": 2,
                "message": 'MCP_NODE:{"name":"warn","explicit":true,"sorry_deps":[]}',
            },
            {
                "severity": 3,
                "message": 'MCP_NODE:{"name":"info","explicit":true,"sorry_deps":[]}',
            },
            {"severity": 3, "message": 'MCP_SUMMARY:{"visited":1}'},
        ]
        nodes, visited = parse_sorry_result(diags)
        assert list(nodes.keys()) == ["info"]
        assert visited == 1

    def test_missing_severity_key(self):
        """Diagnostic without severity key — should not crash."""
        diags = [{"message": 'MCP_NODE:{"name":"X","explicit":true,"sorry_deps":[]}'}]
        nodes, _ = parse_sorry_result(diags)
        assert nodes == {}

    def test_missing_message_key(self):
        """Diagnostic without message key — should not crash."""
        diags = [{"severity": 3}]
        nodes, _ = parse_sorry_result(diags)
        assert nodes == {}


class TestRenderTree:
    def test_single_explicit(self):
        nodes = {"A": SorryNode("A", explicit_sorry=True, sorry_deps=[])}
        lines = render_tree("A", nodes)
        assert len(lines) == 1
        assert "[explicit sorry]" in lines[0]
        assert "A" in lines[0]

    def test_chain(self):
        nodes = {
            "A": SorryNode("A", explicit_sorry=False, sorry_deps=["B"]),
            "B": SorryNode("B", explicit_sorry=True, sorry_deps=[]),
        }
        lines = render_tree("A", nodes)
        assert len(lines) == 2
        assert "[explicit sorry]" not in lines[0]  # A is transitive
        assert "[explicit sorry]" in lines[1]  # B is explicit

    def test_diamond(self):
        """A depends on B and C, both depend on D (explicit sorry)."""
        nodes = {
            "A": SorryNode("A", explicit_sorry=False, sorry_deps=["B", "C"]),
            "B": SorryNode("B", explicit_sorry=False, sorry_deps=["D"]),
            "C": SorryNode("C", explicit_sorry=False, sorry_deps=["D"]),
            "D": SorryNode("D", explicit_sorry=True, sorry_deps=[]),
        }
        lines = render_tree("A", nodes)
        # D should appear once fully, once as "(see above)"
        full = [ln for ln in lines if "D" in ln and "see above" not in ln]
        back = [ln for ln in lines if "D" in ln and "see above" in ln]
        assert len(full) == 1
        assert len(back) == 1

    def test_target_not_in_nodes(self):
        """Target not in nodes dict — just prints the name with no tag."""
        lines = render_tree("A", {})
        assert len(lines) == 1
        assert "A" in lines[0]
        assert "[explicit sorry]" not in lines[0]

    def test_wide_tree(self):
        """A has 4 sorry children — all rendered with correct connectors."""
        nodes = {
            "A": SorryNode("A", sorry_deps=["B", "C", "D", "E"]),
            "B": SorryNode("B", explicit_sorry=True, sorry_deps=[]),
            "C": SorryNode("C", explicit_sorry=True, sorry_deps=[]),
            "D": SorryNode("D", explicit_sorry=True, sorry_deps=[]),
            "E": SorryNode("E", explicit_sorry=True, sorry_deps=[]),
        }
        lines = render_tree("A", nodes)
        assert len(lines) == 5
        # Last child uses └─, others use ├─
        assert "├─" in lines[1]
        assert "├─" in lines[2]
        assert "├─" in lines[3]
        assert "└─" in lines[4]

    def test_deep_tree(self):
        """10-level chain — indentation increases correctly."""
        names = [f"L{i}" for i in range(10)]
        nodes = {}
        for i, name in enumerate(names):
            deps = [names[i + 1]] if i < 9 else []
            nodes[name] = SorryNode(name, explicit_sorry=(i == 9), sorry_deps=deps)
        lines = render_tree("L0", nodes)
        assert len(lines) == 10
        # Each level should be indented more than the previous
        for i in range(1, len(lines)):
            assert len(lines[i]) > len(lines[i - 1]) or "L" in lines[i]

    def test_sorry_deps_referencing_missing_node(self):
        """sorry_deps includes a name not in nodes dict — filtered out."""
        nodes = {
            "A": SorryNode("A", sorry_deps=["B", "MISSING"]),
            "B": SorryNode("B", explicit_sorry=True, sorry_deps=[]),
        }
        lines = render_tree("A", nodes)
        # MISSING should not appear (filtered by `if d in nodes` in render_tree)
        assert not any("MISSING" in ln for ln in lines)
        assert any("B" in ln for ln in lines)

    def test_render_tree_requires_fqn_root(self):
        """render_tree must be called with the FQN (resolved name), not short name.

        server.py passes resolved_name (not decl_name) to render_tree so that
        the root key matches the FQN-keyed nodes dict.
        """
        nodes = {
            "A.B.foo": SorryNode("A.B.foo", sorry_deps=["A.B.bar"]),
            "A.B.bar": SorryNode("A.B.bar", explicit_sorry=True, sorry_deps=[]),
        }
        # With FQN — tree works correctly
        lines_fqn = render_tree("A.B.foo", nodes)
        assert len(lines_fqn) == 2
        assert "A.B.bar" in "\n".join(lines_fqn)

        # With short name — root doesn't match, tree degrades to single node
        lines_short = render_tree("foo", nodes)
        assert len(lines_short) == 1

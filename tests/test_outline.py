"""Test outline generation with various Lean files."""

from __future__ import annotations

import asyncio
import orjson
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from lean_lsp_mcp.outline_utils import (
    _OUTLINE_CACHE,
    _OUTLINE_CACHE_LOCK,
    generate_outline_data,
)
from tests.helpers.mcp_client import MCPClient, result_text


def _parse_outline(result) -> dict:
    """Parse the JSON outline result."""
    text = result_text(result)
    return orjson.loads(text)


class _FakeOutlineClient:
    project_path = "/tmp/lean-lsp-mcp-outline-cache-test"

    def __init__(self) -> None:
        self.document_symbols_calls = 0

    async def reload_from_disk(self, path: str, wait: bool = False):
        self.path = path

    def content(self, path: str) -> str:
        return "import Mathlib\nnamespace CacheNs\nend CacheNs\n"

    async def document_symbols(self, path: str, fresh: bool = True) -> list[dict]:
        self.document_symbols_calls += 1
        return [
            {
                "name": "CacheNs",
                "kind": "namespace",
                "range": {
                    "start": {"line": 1, "character": 0},
                    "end": {"line": 2, "character": 11},
                },
                "children": [],
            }
        ]


class _BlockingOutlineClient:
    project_path = "/tmp/lean-lsp-mcp-outline-race-test"

    def __init__(self) -> None:
        self.namespace = "BeforeEdit"
        self.document_symbols_calls = 0
        self.request_started = asyncio.Event()
        self.release_request = asyncio.Event()

    async def reload_from_disk(self, path: str, wait: bool = False):
        self.path = path

    def content(self, path: str) -> str:
        return f"namespace {self.namespace}\nend {self.namespace}\n"

    async def document_symbols(self, path: str, fresh: bool = True) -> list[dict]:
        namespace = self.namespace
        self.document_symbols_calls += 1
        self.request_started.set()
        await self.release_request.wait()
        return [
            {
                "name": namespace,
                "kind": "namespace",
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 1, "character": len(namespace) + 4},
                },
                "children": [],
            }
        ]


class _LemmaOutlineClient:
    project_path = "/tmp/lean-lsp-mcp-lemma-outline-test"

    def __init__(self) -> None:
        self.source = (
            "@[simp] lemma topLevel : True := trivial\n"
            "namespace Inside\n"
            "lemma nested : True := by\n"
            "  trivial\n"
            "end Inside\n"
        )
        self.requested_fresh_symbols = True

    async def reload_from_disk(self, path: str, wait: bool = False):
        self.path = path

    def content(self, path: str) -> str:
        return self.source

    async def document_symbols(self, path: str, fresh: bool = True) -> list[dict]:
        self.requested_fresh_symbols = fresh
        return [
            {
                "name": "Inside",
                "kind": "namespace",
                "range": {
                    "start": {"line": 1, "character": 0},
                    "end": {"line": 3, "character": 10},
                },
                "children": [
                    {
                        "name": "Inside.nested",
                        "kind": 6,
                        "detail": ": True",
                        "range": {
                            "start": {"line": 2, "character": 0},
                            "end": {"line": 3, "character": 10},
                        },
                        "children": [],
                    }
                ],
            }
        ]


class _KindOutlineClient:
    project_path = "/tmp/lean-lsp-mcp-outline-kinds-test"

    async def reload_from_disk(self, path: str, wait: bool = False):
        self.path = path

    def content(self, path: str) -> str:
        return (
            "namespace KindDemo\n"
            "structure Point where\n"
            "  x : Nat\n"
            "class PointClass where\n"
            "  value : Nat\n"
            "instance : Inhabited Point := ⟨⟨0⟩⟩\n"
            "end KindDemo\n"
        )

    async def document_symbols(self, path: str, fresh: bool = True) -> list[dict]:
        return [
            {
                "name": "KindDemo",
                "kind": "namespace",
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 6, "character": 13},
                },
                "children": [
                    {
                        "name": "KindDemo.Point",
                        "kind": 6,
                        "range": {
                            "start": {"line": 1, "character": 0},
                            "end": {"line": 2, "character": 9},
                        },
                        "children": [],
                    },
                    {
                        "name": "KindDemo.PointClass",
                        "kind": 5,
                        "range": {
                            "start": {"line": 3, "character": 0},
                            "end": {"line": 4, "character": 13},
                        },
                        "children": [],
                    },
                    {
                        "name": "instance : Inhabited Point ",
                        "kind": 6,
                        "range": {
                            "start": {"line": 5, "character": 0},
                            "end": {"line": 5, "character": 39},
                        },
                        "children": [],
                    },
                ],
            }
        ]


@pytest.mark.asyncio
async def test_outline_cache_reuses_unchanged_file_symbols() -> None:
    """Repeated unchanged outlines should avoid another documentSymbol call."""
    with _OUTLINE_CACHE_LOCK:
        _OUTLINE_CACHE.clear()

    client = _FakeOutlineClient()
    first = await generate_outline_data(client, "Cache.lean")
    second = await generate_outline_data(client, "Cache.lean")

    assert client.document_symbols_calls == 1
    assert second == first

    first.declarations[0].name = "mutated"
    third = await generate_outline_data(client, "Cache.lean")
    assert third.declarations[0].name == "CacheNs"


@pytest.mark.asyncio
async def test_outline_cache_does_not_publish_stale_inflight_result() -> None:
    """An old in-flight result must not become the cache entry for new content."""
    with _OUTLINE_CACHE_LOCK:
        _OUTLINE_CACHE.clear()

    client = _BlockingOutlineClient()
    first_task = asyncio.create_task(generate_outline_data(client, "Cache.lean"))
    await client.request_started.wait()
    client.namespace = "AfterEdit"
    client.release_request.set()

    first = await first_task
    second = await generate_outline_data(client, "Cache.lean")

    assert first.declarations[0].name == "BeforeEdit"
    assert second.declarations[0].name == "AfterEdit"
    assert client.document_symbols_calls == 2


@pytest.mark.asyncio
async def test_outline_deduplicates_and_sorts_lemmas_without_barrier() -> None:
    with _OUTLINE_CACHE_LOCK:
        _OUTLINE_CACHE.clear()

    client = _LemmaOutlineClient()
    outline = await generate_outline_data(client, "Lemmas.lean")

    lemmas = [entry for entry in outline.declarations if entry.kind == "Thm"]
    assert [entry.name for entry in lemmas] == ["topLevel", "Inside.nested"]
    assert sum(entry.name.endswith("nested") for entry in lemmas) == 1
    nested = next(entry for entry in lemmas if entry.name.endswith("nested"))
    assert nested.end_line == 4
    assert all(entry.type_signature == ": True" for entry in lemmas)
    assert client.requested_fresh_symbols is False


@pytest.mark.asyncio
async def test_outline_uses_structure_kind_and_cleans_instance_name() -> None:
    with _OUTLINE_CACHE_LOCK:
        _OUTLINE_CACHE.clear()

    outline = await generate_outline_data(_KindOutlineClient(), "Kinds.lean")
    point = next(
        entry for entry in outline.declarations if entry.name.endswith("Point")
    )
    point_class = next(
        entry for entry in outline.declarations if entry.name.endswith("PointClass")
    )
    instance = next(
        entry for entry in outline.declarations if "Inhabited Point" in entry.name
    )

    assert point.kind == "Struct"
    assert point_class.kind == "Class"
    assert instance.name == "instance : Inhabited Point"


@pytest.fixture
def mathlib_nat_basic(test_project_path: Path) -> Path:
    """Path to Mathlib Data.Nat.Basic file."""
    return test_project_path / ".lake/packages/mathlib/Mathlib/Data/Nat/Basic.lean"


@pytest.fixture
def lemma_outline_file(test_project_path: Path) -> Path:
    path = test_project_path / "LemmaOutlineTest.lean"
    path.write_text(
        "lemma topLevelLemma : True := trivial\n\n"
        "namespace OutlineLemmaNs\n"
        "lemma nestedLemma : True := trivial\n"
        "end OutlineLemmaNs\n",
        encoding="utf-8",
    )
    yield path
    path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_outline_includes_top_level_and_namespaced_lemmas(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    lemma_outline_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_file_outline", {"file_path": str(lemma_outline_file)}
        )

    outline = _parse_outline(result)
    lemmas = [
        declaration
        for declaration in outline["declarations"]
        if declaration["kind"] == "Thm"
    ]
    assert {declaration["name"] for declaration in lemmas} >= {
        "topLevelLemma",
        "nestedLemma",
    }
    assert all(declaration.get("type_signature") for declaration in lemmas)


@pytest.mark.asyncio
async def test_outline_simple_files(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """Test outline generation on simple test files."""
    test_files = [
        test_project_path / "StructTest.lean",
        test_project_path / "TheoremTest.lean",
    ]

    # Skip if test files don't exist
    existing_files = [f for f in test_files if f.exists()]
    if not existing_files:
        pytest.skip("Test files StructTest.lean/TheoremTest.lean not found")

    async with mcp_client_factory() as client:
        for test_file in existing_files:
            result = await client.call_tool(
                "lean_file_outline", {"file_path": str(test_file)}
            )
            outline = _parse_outline(result)

            # Basic structure checks - now JSON with imports and declarations
            assert "imports" in outline or "declarations" in outline
            assert isinstance(outline.get("imports", []), list)
            assert isinstance(outline.get("declarations", []), list)


@pytest.mark.asyncio
async def test_mathlib_outline_structure(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    mathlib_nat_basic: Path,
) -> None:
    """Test outline generation with a real Mathlib file."""
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_file_outline", {"file_path": str(mathlib_nat_basic)}
        )
        outline = _parse_outline(result)

        # Basic structure checks - now structured JSON
        assert "imports" in outline
        assert "declarations" in outline

        # Should have imports from Mathlib
        assert any("Mathlib.Data.Nat.Init" in imp for imp in outline["imports"])

        # Should have namespace declarations with kind "Ns"
        decl_names = [d["name"] for d in outline["declarations"]]
        decl_kinds = [d["kind"] for d in outline["declarations"]]
        assert "Nat" in decl_names or any("Ns" in k for k in decl_kinds)

        # Should have instance declarations
        assert any(
            "instLinearOrder" in d["name"]
            or "LinearOrder" in (d.get("type_signature") or "")
            for d in outline["declarations"]
        )
        assert any(
            d["name"] == "succ_injective" and d["kind"] == "Thm"
            for d in outline["declarations"]
        )


@pytest.mark.asyncio
async def test_mathlib_outline_has_line_numbers(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    mathlib_nat_basic: Path,
) -> None:
    """Verify line numbers are present in outline."""
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_file_outline", {"file_path": str(mathlib_nat_basic)}
        )
        outline = _parse_outline(result)

        # All declarations should have start_line and end_line
        for decl in outline["declarations"]:
            assert "start_line" in decl and isinstance(decl["start_line"], int)
            assert "end_line" in decl and isinstance(decl["end_line"], int)
            assert decl["start_line"] > 0
            assert decl["end_line"] >= decl["start_line"]


@pytest.mark.asyncio
async def test_mathlib_outline_has_types(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    mathlib_nat_basic: Path,
) -> None:
    """Verify type signatures are included."""
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_file_outline", {"file_path": str(mathlib_nat_basic)}
        )
        outline = _parse_outline(result)

        # Some declarations should have type_signature
        has_type_sig = any(d.get("type_signature") for d in outline["declarations"])
        assert has_type_sig, "At least some declarations should have type signatures"


@pytest.mark.asyncio
async def test_mathlib_outline_does_not_modify_source_file(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    mathlib_nat_basic: Path,
) -> None:
    """Outline generation reads a Mathlib source file without editing it."""
    async with mcp_client_factory() as client:
        # Get original file content
        original_content = mathlib_nat_basic.read_text()

        # Generate the outline and confirm the original source is untouched.
        await client.call_tool(
            "lean_file_outline", {"file_path": str(mathlib_nat_basic)}
        )

        # Read file content again
        final_content = mathlib_nat_basic.read_text()

        # File should be unchanged
        assert final_content == original_content, "Outline generation changed the file"

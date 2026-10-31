from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import orjson
import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.fixture
def additive_reference_file(test_project_path: Path) -> Path:
    path = test_project_path / "DeclarationSourceTest.lean"
    path.write_text(
        "import Mathlib\n\n#check Finset.sum_range_succ\n", encoding="utf-8"
    )
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def declaration_probe_file(test_project_path: Path) -> Path:
    path = test_project_path / "DeclarationProbeTest.lean"
    path.write_text("import Mathlib\n", encoding="utf-8")
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def qualified_local_declaration_file(test_project_path: Path) -> Path:
    path = test_project_path / "QualifiedLocalDeclarationTest.lean"
    path.write_text(
        "import Mathlib\n\n"
        "namespace Outer\n"
        "section Helpers\n"
        "variable {α : Type*}\n\n"
        "lemma helper (a : α) : a = a := rfl\n\n"
        "namespace Inner\n"
        "theorem deep : True := trivial\n"
        "@[simp] lemma deep_simp (n : ℕ) : n + 0 = n := by simp\n"
        "end Inner\n"
        "end Helpers\n\n"
        "class Greeter (α : Type) where\n"
        "  greet : α → String\n\n"
        'instance instGreeterNat : Greeter ℕ := ⟨fun n => s!"hi {n}"⟩\n\n'
        "private def secret : ℕ := 7\n"
        "end Outer\n",
        encoding="utf-8",
    )
    yield path
    path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_declaration_file_reports_additive_source_and_location(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    additive_reference_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_declaration_file",
            {
                "file_path": str(additive_reference_file),
                "symbol": "Finset.sum_range_succ",
                "context_lines": 0,
            },
        )

    declaration = orjson.loads(result_text(result))
    assert declaration["declaration_start_line"] is not None
    assert (
        declaration["declaration_end_line"] is None
        or declaration["declaration_end_line"] >= declaration["declaration_start_line"]
    )
    assert declaration["start_line"] == declaration["declaration_start_line"]
    assert "prod_range_succ" in declaration["content"]
    assert declaration["source_declaration"] == "Finset.prod_range_succ"


@pytest.mark.asyncio
async def test_declaration_file_resolves_name_missing_from_source(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    declaration_probe_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_declaration_file",
            {
                "file_path": str(declaration_probe_file),
                "symbol": "Nat.add_comm",
                "context_lines": 0,
            },
        )

    declaration = orjson.loads(result_text(result))
    assert declaration["declaration_start_line"] is not None
    assert declaration["file_path"].endswith(".lean")
    assert "add_comm" in declaration["content"]


@pytest.mark.asyncio
async def test_declaration_file_resolves_qualified_local_name(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    qualified_local_declaration_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_declaration_file",
            {
                "file_path": str(qualified_local_declaration_file),
                "symbol": "Outer.Inner.deep_simp",
                "context_lines": 0,
            },
        )

    declaration = orjson.loads(result_text(result))
    assert declaration["file_path"] == qualified_local_declaration_file.name
    assert declaration["declaration_start_line"] == 11
    assert "lemma deep_simp" in declaration["content"]

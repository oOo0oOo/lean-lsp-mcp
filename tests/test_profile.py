"""Integration tests for lean_profile_proof tool.

These tests run actual Lean profiling and verify the output structure.
If Lean's profiler format changes, these tests should fail.
"""

from pathlib import Path

import pytest

from lean_lsp_mcp.profile_utils import profile_theorem
from tests.helpers.mcp_client import MCPToolError, result_json


class TestProfileTheorem:
    """Direct API tests - run real Lean profiling."""

    @pytest.fixture
    def profile_file(self, test_project_path: Path) -> Path:
        return test_project_path / "ProfileTest.lean"

    @pytest.mark.asyncio
    async def test_profiles_rw_theorem(self, profile_file: Path):
        """Line 3: theorem simple_by using rw tactic."""
        profile = await profile_theorem(
            profile_file, theorem_line=3, project_path=profile_file.parent
        )
        assert profile.ms > 0
        assert len(profile.categories) > 0
        # Should extract timing for line 4 (rw)
        if profile.lines:
            ln = profile.lines[0]
            assert ln.line == 4  # rw is on line 4
            assert "rw" in ln.text

    @pytest.mark.asyncio
    async def test_profiles_simp_theorem(self, profile_file: Path):
        """Line 6: theorem simp_test using simp tactic."""
        profile = await profile_theorem(
            profile_file, theorem_line=6, project_path=profile_file.parent
        )
        assert profile.ms > 0
        assert "simp" in profile.categories
        if profile.lines:
            assert "simp" in profile.lines[0].text

    @pytest.mark.asyncio
    async def test_profiles_omega_theorem(self, profile_file: Path):
        """Line 9: theorem omega_test using omega tactic."""
        profile = await profile_theorem(
            profile_file, theorem_line=9, project_path=profile_file.parent
        )
        assert profile.ms > 0
        if profile.lines:
            ln = profile.lines[0]
            assert ln.line == 10  # omega is on line 10
            assert "omega" in ln.text
            assert ln.ms > 0

    @pytest.mark.asyncio
    async def test_invalid_line_raises(self, profile_file: Path):
        with pytest.raises(ValueError):
            await profile_theorem(
                profile_file, theorem_line=999, project_path=profile_file.parent
            )


class TestProfileProofTool:
    """MCP tool tests."""

    @pytest.mark.asyncio
    async def test_returns_structured_profile(
        self, mcp_client_factory, test_project_path: Path
    ):
        async with mcp_client_factory() as client:
            result = await client.call_tool(
                "lean_profile_proof",
                {
                    "file_path": str(test_project_path / "ProfileTest.lean"),
                    "line": 6,
                },
            )
            data = result_json(result)
            assert data["ms"] > 0
            assert "simp" in data["categories"]
            # Verify line structure
            if data["lines"]:
                ln = data["lines"][0]
                assert "text" in ln and "ms" in ln and "line" in ln

    @pytest.mark.asyncio
    async def test_error_on_invalid_line(
        self, mcp_client_factory, test_project_path: Path
    ):
        async with mcp_client_factory() as client:
            with pytest.raises(MCPToolError):
                await client.call_tool(
                    "lean_profile_proof",
                    {
                        "file_path": str(test_project_path / "ProfileTest.lean"),
                        "line": 999,
                    },
                )

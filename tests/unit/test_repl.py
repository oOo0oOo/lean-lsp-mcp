"""Unit tests for REPL manager and models."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lean_lsp_mcp.repl import ReplManager, ReplSession, get_repl_cache_dir
from lean_lsp_mcp.repl_models import (
    ReplCommandRequest,
    ReplCommandResponse,
    ReplCmdResult,
    ReplMessage,
    ReplPosition,
    ReplSessionInfo,
    ReplSorry,
    ReplTacticRequest,
    ReplTacticResponse,
    ReplTacticResult,
)


class TestGetCacheDir:
    """Tests for cache directory resolution."""

    def test_default_cache_dir(self, monkeypatch):
        """Default to ~/.cache/lean-lsp-mcp/repl."""
        monkeypatch.delenv("LEAN_REPL_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        cache_dir = get_repl_cache_dir()
        assert cache_dir == Path.home() / ".cache" / "lean-lsp-mcp" / "repl"

    def test_env_override(self, monkeypatch):
        """LEAN_REPL_CACHE_DIR takes precedence."""
        monkeypatch.setenv("LEAN_REPL_CACHE_DIR", "/custom/cache")
        cache_dir = get_repl_cache_dir()
        assert cache_dir == Path("/custom/cache")

    def test_xdg_cache_home(self, monkeypatch):
        """XDG_CACHE_HOME is respected."""
        monkeypatch.delenv("LEAN_REPL_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", "/xdg/cache")
        cache_dir = get_repl_cache_dir()
        assert cache_dir == Path("/xdg/cache") / "lean-lsp-mcp" / "repl"


class TestReplSession:
    """Tests for ReplSession state management."""

    def test_session_creation(self, tmp_path):
        """Session initializes with correct defaults."""
        session = ReplSession(
            session_id="test-123",
            project_path=tmp_path,
        )
        assert session.session_id == "test-123"
        assert session.project_path == tmp_path
        assert session.current_env is None
        assert session.proof_states == {}
        assert session.created_at > 0

    def test_update_env(self, tmp_path):
        """update_env tracks environment IDs."""
        session = ReplSession(session_id="test", project_path=tmp_path)
        assert session.current_env is None

        session.update_env(1)
        assert session.current_env == 1

        session.update_env(5)
        assert session.current_env == 5

        # None doesn't overwrite
        session.update_env(None)
        assert session.current_env == 5

    def test_add_proof_states(self, tmp_path):
        """add_proof_states tracks sorries."""
        session = ReplSession(session_id="test", project_path=tmp_path)

        sorries = [
            ReplSorry(
                pos=ReplPosition(line=1, column=1),
                goal="⊢ 1 + 1 = 2",
                proofState=0,
            ),
            ReplSorry(
                pos=ReplPosition(line=5, column=10),
                goal="⊢ True",
                proofState=1,
            ),
        ]
        session.add_proof_states(sorries)

        assert 0 in session.proof_states
        assert 1 in session.proof_states
        assert session.proof_states[0]["goal"] == "⊢ 1 + 1 = 2"
        assert session.proof_states[1]["goal"] == "⊢ True"


class TestReplManagerSessionManagement:
    """Tests for ReplManager session operations."""

    def test_create_session(self, tmp_path):
        """create_session generates unique sessions."""
        manager = ReplManager(cache_dir=tmp_path / "cache")
        project = tmp_path / "project"
        project.mkdir()

        session1 = manager.create_session(project)
        session2 = manager.create_session(project)

        assert session1.session_id != session2.session_id
        assert session1.project_path == project
        assert session2.project_path == project

    def test_get_session(self, tmp_path):
        """get_session retrieves by ID."""
        manager = ReplManager(cache_dir=tmp_path / "cache")
        project = tmp_path / "project"
        project.mkdir()

        session = manager.create_session(project)
        retrieved = manager.get_session(session.session_id)

        assert retrieved is session
        assert manager.get_session("nonexistent") is None

    def test_list_sessions(self, tmp_path):
        """list_sessions filters by project."""
        manager = ReplManager(cache_dir=tmp_path / "cache")
        project1 = tmp_path / "project1"
        project2 = tmp_path / "project2"
        project1.mkdir()
        project2.mkdir()

        s1 = manager.create_session(project1)
        s2 = manager.create_session(project1)
        s3 = manager.create_session(project2)

        all_sessions = manager.list_sessions()
        assert len(all_sessions) == 3

        p1_sessions = manager.list_sessions(project1)
        assert len(p1_sessions) == 2
        assert s1 in p1_sessions
        assert s2 in p1_sessions

        p2_sessions = manager.list_sessions(project2)
        assert len(p2_sessions) == 1
        assert s3 in p2_sessions

    def test_delete_session(self, tmp_path):
        """delete_session removes session."""
        manager = ReplManager(cache_dir=tmp_path / "cache")
        project = tmp_path / "project"
        project.mkdir()

        session = manager.create_session(project)
        session_id = session.session_id

        assert manager.get_session(session_id) is not None
        assert manager.delete_session(session_id) is True
        assert manager.get_session(session_id) is None
        assert manager.delete_session(session_id) is False  # Already deleted

    def test_default_session(self, tmp_path):
        """get_or_create_default_session creates once per project."""
        manager = ReplManager(cache_dir=tmp_path / "cache")
        project = tmp_path / "project"
        project.mkdir()

        default1 = manager.get_or_create_default_session(project)
        default2 = manager.get_or_create_default_session(project)

        assert default1 is default2
        assert "_default:" in default1.session_id


class TestReplManagerBinaryDiscovery:
    """Tests for REPL binary discovery."""

    def test_project_local_binary(self, tmp_path):
        """Project-local binary is preferred."""
        manager = ReplManager(cache_dir=tmp_path / "cache")
        project = tmp_path / "project"
        local_repl = project / ".lake" / "build" / "bin" / "repl"
        local_repl.parent.mkdir(parents=True)
        local_repl.touch()

        binary = manager._get_project_repl_binary(project)
        assert binary == local_repl

    def test_no_project_binary(self, tmp_path):
        """Returns None when no project-local binary."""
        manager = ReplManager(cache_dir=tmp_path / "cache")
        project = tmp_path / "project"
        project.mkdir()

        binary = manager._get_project_repl_binary(project)
        assert binary is None

    def test_global_binary_path(self, tmp_path):
        """global_binary_path returns correct location."""
        manager = ReplManager(cache_dir=tmp_path / "cache")
        expected = tmp_path / "cache" / "repo" / ".lake" / "build" / "bin" / "repl"
        assert manager.global_binary_path == expected

    def test_is_global_installed(self, tmp_path):
        """is_global_installed checks binary exists."""
        manager = ReplManager(cache_dir=tmp_path / "cache")
        assert manager.is_global_installed is False

        manager.global_binary_path.parent.mkdir(parents=True)
        manager.global_binary_path.touch()
        assert manager.is_global_installed is True


class TestReplModels:
    """Tests for Pydantic model serialization."""

    def test_command_request_serialization(self):
        """ReplCommandRequest serializes correctly."""
        req = ReplCommandRequest(cmd="def x := 1")
        data = req.model_dump(exclude_none=True)
        assert data == {"cmd": "def x := 1"}

        req_with_env = ReplCommandRequest(cmd="def y := 2", env=1)
        data_with_env = req_with_env.model_dump(exclude_none=True)
        assert data_with_env == {"cmd": "def y := 2", "env": 1}

    def test_tactic_request_serialization(self):
        """ReplTacticRequest serializes correctly."""
        req = ReplTacticRequest(tactic="rfl", proofState=0)
        data = req.model_dump(exclude_none=True)
        assert data == {"tactic": "rfl", "proofState": 0}

    def test_command_response_parsing(self):
        """ReplCommandResponse parses from dict."""
        data = {
            "env": 1,
            "sorries": [
                {"pos": {"line": 1, "column": 10}, "goal": "⊢ True", "proofState": 0}
            ],
            "messages": [],
        }
        response = ReplCommandResponse.model_validate(data)
        assert response.env == 1
        assert len(response.sorries) == 1
        assert response.sorries[0].proofState == 0

    def test_tactic_response_parsing(self):
        """ReplTacticResponse parses from dict."""
        data = {
            "proofState": 1,
            "goals": ["⊢ 1 = 1"],
            "messages": [],
        }
        response = ReplTacticResponse.model_validate(data)
        assert response.proofState == 1
        assert response.goals == ["⊢ 1 = 1"]

    def test_cmd_result_model(self):
        """ReplCmdResult wraps response correctly."""
        result = ReplCmdResult(
            env=1,
            sorries=[
                ReplSorry(
                    pos=ReplPosition(line=1, column=1),
                    goal="⊢ P",
                    proofState=0,
                )
            ],
            messages=[],
            success=True,
        )
        assert result.env == 1
        assert result.success is True
        assert len(result.sorries) == 1


class TestReplManagerResponseParsing:
    """Tests for response parsing in ReplManager."""

    def test_parse_command_response_success(self, tmp_path):
        """_parse_command_response handles success."""
        manager = ReplManager(cache_dir=tmp_path)
        data = {
            "env": 2,
            "sorries": [
                {"pos": {"line": 5, "column": 8}, "goal": "⊢ n = n", "proofState": 3}
            ],
            "messages": [
                {"severity": "warning", "pos": {"line": 1, "column": 1}, "data": "unused variable"}
            ],
        }
        response = manager._parse_command_response(data)

        assert response.env == 2
        assert len(response.sorries) == 1
        assert response.sorries[0].proofState == 3
        assert len(response.messages) == 1
        assert response.messages[0].severity == "warning"

    def test_parse_command_response_error(self, tmp_path):
        """_parse_command_response handles errors."""
        manager = ReplManager(cache_dir=tmp_path)
        data = {"error": "Unknown environment 99"}
        response = manager._parse_command_response(data)

        assert response.env is None
        assert len(response.messages) == 1
        assert response.messages[0].severity == "error"
        assert "Unknown environment" in response.messages[0].data

    def test_parse_tactic_response_success(self, tmp_path):
        """_parse_tactic_response handles success."""
        manager = ReplManager(cache_dir=tmp_path)
        data = {
            "proofState": 4,
            "goals": ["⊢ a = a", "⊢ b = b"],
        }
        response = manager._parse_tactic_response(data)

        assert response.proofState == 4
        assert len(response.goals) == 2

    def test_parse_tactic_response_complete(self, tmp_path):
        """_parse_tactic_response handles proof completion."""
        manager = ReplManager(cache_dir=tmp_path)
        data = {
            "proofState": None,
            "goals": [],
        }
        response = manager._parse_tactic_response(data)

        assert response.proofState is None
        assert response.goals == []

    def test_serialize_request(self, tmp_path):
        """_serialize_request produces valid JSON."""
        manager = ReplManager(cache_dir=tmp_path)

        req1 = ReplCommandRequest(cmd="def x := 1")
        json1 = manager._serialize_request(req1)
        assert json1.strip() == '{"cmd":"def x := 1"}'

        req2 = ReplCommandRequest(cmd="def y := 2", env=5)
        json2 = manager._serialize_request(req2)
        assert '"env":5' in json2
        assert '"cmd":"def y := 2"' in json2

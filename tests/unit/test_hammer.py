"""Tests for hammer module."""

from __future__ import annotations

import shutil

import pytest

from lean_lsp_mcp.hammer import HammerManager, get_cache_dir


class TestGetCacheDir:
    def test_default_cache_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEAN_HAMMER_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        cache_dir = get_cache_dir()
        assert "lean-lsp-mcp" in str(cache_dir)
        assert "hammer" in str(cache_dir)

    def test_custom_cache_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEAN_HAMMER_CACHE_DIR", "/custom/path")
        cache_dir = get_cache_dir()
        assert str(cache_dir) == "/custom/path"


class TestHammerManager:
    def test_default_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEAN_HAMMER_PORT", raising=False)
        manager = HammerManager()
        assert manager.port == 8765

    def test_custom_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEAN_HAMMER_PORT", "9000")
        manager = HammerManager()
        assert manager.port == 9000

    def test_url_property(self) -> None:
        manager = HammerManager(port=8080)
        assert manager.url == "http://localhost:8080"

    def test_custom_image_and_container_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LEAN_HAMMER_IMAGE", "example/hammer:test")
        monkeypatch.setenv("LEAN_HAMMER_CONTAINER_NAME", "custom-hammer")
        manager = HammerManager()
        assert manager.image == "example/hammer:test"
        assert manager.container_name == "custom-hammer"

    def test_find_container_tool_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_which(cmd: str) -> str | None:
            if cmd == "docker":
                return "/usr/bin/docker"
            return None

        monkeypatch.setattr(shutil, "which", mock_which)
        manager = HammerManager()
        assert manager._find_container_tool() == "docker"

    def test_find_container_tool_container(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def mock_which(cmd: str) -> str | None:
            if cmd == "container":
                return "/usr/local/bin/container"
            return None

        monkeypatch.setattr(shutil, "which", mock_which)
        manager = HammerManager()
        assert manager._find_container_tool() == "container"

    def test_find_container_tool_prefers_container(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def mock_which(cmd: str) -> str | None:
            # Both are available
            if cmd in ("container", "docker"):
                return f"/usr/bin/{cmd}"
            return None

        monkeypatch.setattr(shutil, "which", mock_which)
        manager = HammerManager()
        # Should prefer macOS container tool
        assert manager._find_container_tool() == "container"

    def test_find_container_tool_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda x: None)
        manager = HammerManager()
        assert manager._find_container_tool() is None

    def test_is_available_with_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            shutil, "which", lambda x: "/usr/bin/docker" if x == "docker" else None
        )
        manager = HammerManager()
        assert manager.is_available() is True

    def test_is_available_without_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda x: None)
        manager = HammerManager()
        assert manager.is_available() is False

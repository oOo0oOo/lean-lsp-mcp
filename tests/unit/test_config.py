"""Tests for the centralized config module."""

import pytest

from lean_lsp_mcp import config


def test_active_transport_default_and_normalization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(config.ACTIVE_TRANSPORT_ENV, raising=False)
    assert config.active_transport() == "stdio"
    monkeypatch.setenv(config.ACTIVE_TRANSPORT_ENV, "  STDIO  ")
    assert config.active_transport() == "stdio"
    monkeypatch.setenv(config.ACTIVE_TRANSPORT_ENV, "")
    assert config.active_transport() == "stdio"


def test_max_open_files(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(config.MAX_OPEN_FILES_ENV, raising=False)
    assert config.max_open_files() == 4
    monkeypatch.setenv(config.MAX_OPEN_FILES_ENV, "10")
    assert config.max_open_files() == 10
    monkeypatch.setenv(config.MAX_OPEN_FILES_ENV, "not-an-int")
    assert config.max_open_files() == 4
    monkeypatch.setenv(config.MAX_OPEN_FILES_ENV, "0")
    assert config.max_open_files() == 4


def test_idle_timeout_seconds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(config.IDLE_TIMEOUT_SECONDS_ENV, raising=False)
    assert config.idle_timeout_seconds() == 600.0
    monkeypatch.setenv(config.IDLE_TIMEOUT_SECONDS_ENV, "30")
    assert config.idle_timeout_seconds() == 30.0
    monkeypatch.setenv(config.IDLE_TIMEOUT_SECONDS_ENV, "0")
    assert config.idle_timeout_seconds() is None
    monkeypatch.setenv(config.IDLE_TIMEOUT_SECONDS_ENV, "not-a-number")
    assert config.idle_timeout_seconds() == 600.0
    monkeypatch.setenv(config.IDLE_TIMEOUT_SECONDS_ENV, "NaN")
    assert config.idle_timeout_seconds() == 600.0
    monkeypatch.setenv(config.IDLE_TIMEOUT_SECONDS_ENV, "inf")
    assert config.idle_timeout_seconds() == 600.0


def test_build_concurrency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(config.BUILD_CONCURRENCY_ENV, raising=False)
    assert config.build_concurrency() == "allow"
    monkeypatch.setenv(config.BUILD_CONCURRENCY_ENV, "CANCEL")
    assert config.build_concurrency() == "cancel"
    monkeypatch.setenv(config.BUILD_CONCURRENCY_ENV, "bogus")
    assert config.build_concurrency() == "allow"


@pytest.mark.parametrize(
    "value,expected",
    [("true", True), ("1", True), ("yes", True), ("", False), ("no", False)],
)
def test_truthy_flags(monkeypatch: pytest.MonkeyPatch, value, expected):
    monkeypatch.setenv(config.LOOGLE_LOCAL_ENV, value)
    assert config.loogle_local_enabled() is expected
    monkeypatch.setenv(config.REPL_ENV, value)
    assert config.repl_enabled() is expected


def test_url_defaults(monkeypatch: pytest.MonkeyPatch):
    for getter, env, default in [
        (config.loogle_url, config.LOOGLE_URL_ENV, config.DEFAULT_LOOGLE_URL),
        (
            config.state_search_url,
            config.STATE_SEARCH_URL_ENV,
            config.DEFAULT_STATE_SEARCH_URL,
        ),
        (config.hammer_url, config.HAMMER_URL_ENV, config.DEFAULT_HAMMER_URL),
    ]:
        monkeypatch.delenv(env, raising=False)
        assert getter() == default
        monkeypatch.setenv(env, "http://localhost:9000")
        assert getter() == "http://localhost:9000"


def test_is_custom_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(config.STATE_SEARCH_URL_ENV, raising=False)
    assert (
        config.is_custom_backend(
            config.STATE_SEARCH_URL_ENV, config.DEFAULT_STATE_SEARCH_URL
        )
        is False
    )
    monkeypatch.setenv(
        config.STATE_SEARCH_URL_ENV, config.DEFAULT_STATE_SEARCH_URL + "/"
    )
    assert (
        config.is_custom_backend(
            config.STATE_SEARCH_URL_ENV, config.DEFAULT_STATE_SEARCH_URL
        )
        is False
    )
    monkeypatch.setenv(config.STATE_SEARCH_URL_ENV, "http://localhost:8000")
    assert (
        config.is_custom_backend(
            config.STATE_SEARCH_URL_ENV, config.DEFAULT_STATE_SEARCH_URL
        )
        is True
    )


def test_repl_numeric_getters(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(config.REPL_TIMEOUT_ENV, raising=False)
    monkeypatch.delenv(config.REPL_MEM_MB_ENV, raising=False)
    assert config.repl_timeout() == 60
    assert config.repl_mem_mb() == 8192
    monkeypatch.setenv(config.REPL_TIMEOUT_ENV, "120")
    assert config.repl_timeout() == 120

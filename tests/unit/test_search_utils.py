import importlib

import pytest


@pytest.fixture(autouse=True)
def reload_search_utils():
    # Ensure a clean module state for each test once the module exists.
    import lean_lsp_mcp.search_utils as search_utils

    importlib.reload(search_utils)
    return search_utils


def test_check_ripgrep_status_when_rg_available(monkeypatch, reload_search_utils):
    search_utils = reload_search_utils
    monkeypatch.setattr(search_utils.shutil, "which", lambda _: "/usr/bin/rg")

    available, message = search_utils.check_ripgrep_status()

    assert available is True
    assert message == ""


@pytest.mark.parametrize(
    "platform_name, expected_snippets",
    [
        (
            "Windows",
            [
                "winget install BurntSushi.ripgrep.MSVC",
                "choco install ripgrep",
            ],
        ),
        (
            "Darwin",
            [
                "brew install ripgrep",
            ],
        ),
        (
            "Linux",
            [
                "sudo apt-get install ripgrep",
                "sudo dnf install ripgrep",
            ],
        ),
        (
            "FreeBSD",
            [
                "Check alternative installation methods.",
            ],
        ),
    ],
)
def test_check_ripgrep_status_when_rg_missing_platform_specific(
    monkeypatch, reload_search_utils, platform_name, expected_snippets
):
    search_utils = reload_search_utils

    monkeypatch.setattr(search_utils.shutil, "which", lambda _: None)
    monkeypatch.setattr(search_utils.platform, "system", lambda: platform_name)

    available, message = search_utils.check_ripgrep_status()

    assert available is False
    assert "ripgrep (rg) was not found on your PATH" in message
    assert "https://github.com/BurntSushi/ripgrep#installation" in message

    for snippet in expected_snippets:
        assert snippet in message

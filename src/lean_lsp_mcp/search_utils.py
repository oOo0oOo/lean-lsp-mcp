"""Utilities for Lean search tools."""

from __future__ import annotations

import platform
import shutil
from typing import Tuple


INSTALL_URL = "https://github.com/BurntSushi/ripgrep#installation"


def check_ripgrep_status() -> Tuple[bool, str]:
    """Check whether ``rg`` is available on PATH and return status + message.

    Returns ``(True, "")`` if the executable is present. Otherwise returns
    ``(False, message)`` where the message explains why ripgrep is required and
    how to install it on the current platform.
    """

    if shutil.which("rg"):
        return True, ""

    system = platform.system()

    base_message = (
        "ripgrep (rg) was not found on your PATH. The lean_local_search tool uses ripgrep for fast declaration search."
    )

    instructions = [
        "\nInstallation options:",
    ]

    if system == "Windows":
        instructions.extend(
            [
                "  - winget install BurntSushi.ripgrep.MSVC",
                "  - choco install ripgrep",
            ]
        )
    elif system == "Darwin":
        instructions.append("  - brew install ripgrep")
    elif system == "Linux":
        instructions.extend(
            [
                "  - sudo apt-get install ripgrep",
                "  - sudo dnf install ripgrep",
            ]
        )
    else:
        instructions.append("  - Check alternative installation methods.")

    instructions.append(f"More installation options: {INSTALL_URL}")

    message = base_message + "\n" + "\n".join(instructions)

    return False, message

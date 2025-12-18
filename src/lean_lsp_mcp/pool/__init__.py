"""REPL pool package for efficient multi-attempt execution.

Adapted from kimina-lean-server (MIT licensed).
Original: https://github.com/project-numina/kimina-lean-server

This package provides:
- Manager: Pool of REPL subprocesses with header-based caching
- Repl: Individual REPL subprocess wrapper
- split_snippet: Header/body splitting for efficient caching
"""

from .manager import Manager, NoAvailableReplError, SnippetResult
from .repl import CommandResponse, Repl, ReplError, ReplResponse
from .settings import PoolSettings, pool_settings
from .split import SplitSnippet, split_snippet

__all__ = [
    "Manager",
    "NoAvailableReplError",
    "SnippetResult",
    "Repl",
    "ReplError",
    "ReplResponse",
    "CommandResponse",
    "PoolSettings",
    "pool_settings",
    "SplitSnippet",
    "split_snippet",
]

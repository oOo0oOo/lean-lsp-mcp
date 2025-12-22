"""REPL pool settings configuration.

Adapted from kimina-lean-server (MIT licensed).

Environment variables:
    LEAN_REPL - Enable REPL pooling (default: true if project path set)
    LEAN_REPL_WORKERS - Max concurrent REPL workers (default: min(threads-2, 6))
    LEAN_REPL_TIMEOUT - Command timeout in seconds (default: 60)
    LEAN_REPL_MEM - Max memory per worker, e.g. "8G" or "4096M" (default: 8G)
"""

import os
import re
from pathlib import Path
from typing import cast

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_workers() -> int:
    """Default worker count: min(threads - 2, 6), minimum 1."""
    threads = os.cpu_count() or 1
    return max(1, min(threads - 2, 6))


class PoolSettings(BaseSettings):
    """Configuration for the REPL pool."""

    # Pool sizing (renamed from max_repls to workers for clarity)
    workers: int = _default_workers()
    max_repl_uses: int = -1  # -1 = unlimited
    mem: int = 8 * 1024  # MB (8GB default)
    max_wait: int = 60  # seconds to wait for worker
    timeout: int = 60  # seconds per REPL command

    # Pre-initialized REPLs by header
    init_repls: dict[str, int] = {}

    # Paths (set at runtime)
    repl_path: Path | None = None
    project_dir: Path | None = None

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="LEAN_REPL_"
    )

    @field_validator("mem", mode="before")
    @classmethod
    def _parse_mem(cls, v: str | int) -> int:
        if isinstance(v, int):
            return cast(int, v * 1024)
        m = re.fullmatch(r"(\d+)([MmGg])", v)
        if m:
            n, unit = m.groups()
            n = int(n)
            return n if unit.lower() == "m" else n * 1024
        raise ValueError("mem must be an int or '<number>[M|G]'")

    @field_validator("workers", mode="before")
    @classmethod
    def _parse_workers(cls, v: int | str) -> int:
        if isinstance(v, str) and v.strip() == "":
            return _default_workers()
        return int(v)


# Global settings instance
pool_settings = PoolSettings()

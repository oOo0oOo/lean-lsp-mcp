"""Pool settings configuration.

Adapted from kimina-lean-server (MIT licensed).
"""

import os
import re
from pathlib import Path
from typing import cast

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PoolSettings(BaseSettings):
    """Configuration for the REPL pool."""

    # Pool sizing
    max_repls: int = max((os.cpu_count() or 1) - 1, 1)
    max_repl_uses: int = -1  # -1 = unlimited
    max_repl_mem: int = 8 * 1024  # MB (8GB default)
    max_wait: int = 60  # seconds

    # Pre-initialized REPLs by header
    init_repls: dict[str, int] = {}

    # Paths (set at runtime)
    repl_path: Path | None = None
    project_dir: Path | None = None

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="LEAN_MCP_"
    )

    @field_validator("max_repl_mem", mode="before")
    @classmethod
    def _parse_max_mem(cls, v: str | int) -> int:
        if isinstance(v, int):
            return cast(int, v * 1024)
        m = re.fullmatch(r"(\d+)([MmGg])", v)
        if m:
            n, unit = m.groups()
            n = int(n)
            return n if unit.lower() == "m" else n * 1024
        raise ValueError("max_repl_mem must be an int or '<number>[M|G]'")

    @field_validator("max_repls", mode="before")
    @classmethod
    def _parse_max_repls(cls, v: int | str) -> int:
        if isinstance(v, str) and v.strip() == "":
            return os.cpu_count() or 1
        return cast(int, v)


# Global settings instance
pool_settings = PoolSettings()

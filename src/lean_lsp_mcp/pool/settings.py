"""Pool configuration from environment variables."""

import os
from dataclasses import dataclass


def _default_workers() -> int:
    threads = os.cpu_count() or 1
    return max(1, min(threads - 2, 6))


def _parse_mem(val: str) -> int:
    """Parse memory string like '4G' or '512M' to MB."""
    val = val.strip().upper()
    if val.endswith("G"):
        return int(val[:-1]) * 1024
    if val.endswith("M"):
        return int(val[:-1])
    return int(val)


@dataclass
class PoolSettings:
    workers: int
    timeout: int
    mem_mb: int
    repl_path: str

    @classmethod
    def from_env(cls) -> "PoolSettings":
        return cls(
            workers=int(os.environ.get("LEAN_REPL_WORKERS", _default_workers())),
            timeout=int(os.environ.get("LEAN_REPL_TIMEOUT", "60")),
            mem_mb=_parse_mem(os.environ.get("LEAN_REPL_MEM", "8G")),
            repl_path=os.environ.get("LEAN_REPL_PATH", "repl"),
        )


def pool_enabled() -> bool:
    return os.environ.get("LEAN_REPL", "").lower() in ("1", "true", "yes")

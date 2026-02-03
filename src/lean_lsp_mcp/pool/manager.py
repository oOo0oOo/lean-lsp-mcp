"""REPL pool manager with header caching."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from .repl import Repl, ReplError
from .settings import PoolSettings
from .split import split_code

logger = logging.getLogger(__name__)


@dataclass
class SnippetResult:
    goals: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class PoolManager:
    """Manages a pool of REPL subprocesses with header caching."""

    def __init__(self, project_dir: str, settings: PoolSettings | None = None):
        self.settings = settings or PoolSettings.from_env()
        self.project_dir = project_dir
        self._free: list[Repl] = []
        self._busy: set[Repl] = set()
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)

    async def _get_repl(self, header: str) -> Repl:
        """Get a REPL with matching header: reuse -> create -> evict."""
        async with self._cond:
            # Reuse existing with matching header
            for i, r in enumerate(self._free):
                if r.header == header:
                    repl = self._free.pop(i)
                    self._busy.add(repl)
                    return repl

            # Create new if under capacity
            total = len(self._free) + len(self._busy)
            if total < self.settings.workers:
                repl = Repl(
                    header,
                    self.settings.repl_path,
                    self.project_dir,
                    self.settings.mem_mb,
                )
                self._busy.add(repl)
                return repl

            # Evict oldest free
            if self._free:
                oldest = self._free.pop(0)
                asyncio.create_task(oldest.close())
                repl = Repl(
                    header,
                    self.settings.repl_path,
                    self.project_dir,
                    self.settings.mem_mb,
                )
                self._busy.add(repl)
                return repl

            # Wait for one to become free
            await self._cond.wait()
            return await self._get_repl(header)

    async def _release(self, repl: Repl) -> None:
        async with self._cond:
            self._busy.discard(repl)
            self._free.append(repl)
            self._cond.notify()

    async def _destroy(self, repl: Repl) -> None:
        async with self._cond:
            self._busy.discard(repl)
            if repl in self._free:
                self._free.remove(repl)
            self._cond.notify()
        await repl.close()

    async def run_multi_attempt(
        self,
        base_code: str,
        snippets: list[str],
        timeout: float | None = None,
    ) -> list[SnippetResult]:
        """Run multiple snippets from same base context with backtracking."""
        timeout = timeout or self.settings.timeout
        split = split_code(base_code)

        repl = await self._get_repl(split.header)
        try:
            # Start and run header if needed
            if not repl.is_running:
                await asyncio.wait_for(repl.start(), timeout=timeout)
                if split.header:
                    resp = await asyncio.wait_for(
                        repl.send(split.header), timeout=timeout
                    )
                    repl.header_env = resp.env

            # Run body to get base env
            base_env = repl.header_env
            if split.body.strip():
                resp = await asyncio.wait_for(
                    repl.send(split.body, env=base_env, gc=base_env is not None),
                    timeout=timeout,
                )
                if resp.env is not None:
                    base_env = resp.env

            # Try each snippet from base_env (true backtracking)
            results: list[SnippetResult] = []
            for snippet in snippets:
                try:
                    resp = await asyncio.wait_for(
                        repl.send(snippet.rstrip(), env=base_env, gc=True),
                        timeout=timeout,
                    )
                    goals = [s.get("goal", "") for s in resp.sorries]
                    results.append(
                        SnippetResult(
                            goals=goals, messages=resp.messages, error=resp.error
                        )
                    )
                except Exception as e:
                    results.append(SnippetResult(error=str(e)))

            await self._release(repl)
            return results

        except Exception as e:
            await self._destroy(repl)
            raise ReplError(f"Pool error: {e}") from e

    async def cleanup(self) -> None:
        async with self._cond:
            repls = self._free + list(self._busy)
            self._free.clear()
            self._busy.clear()
        await asyncio.gather(*(r.close() for r in repls), return_exceptions=True)

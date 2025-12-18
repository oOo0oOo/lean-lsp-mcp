"""REPL pool manager.

Adapted from kimina-lean-server (MIT licensed).
Original: https://github.com/project-numina/kimina-lean-server
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from time import time

from .repl import Repl, ReplError, ReplResponse, close_verbose, is_blank
from .settings import pool_settings

logger = logging.getLogger(__name__)


class NoAvailableReplError(Exception):
    """Raised when no REPL is available within timeout."""

    pass


class Manager:
    """Manages a pool of REPL subprocesses.

    Uses a three-phase allocation strategy:
    1. Reuse: Check free list for matching header
    2. Create: Spawn new if under capacity
    3. Evict: Destroy oldest free REPL if at capacity
    """

    def __init__(
        self,
        *,
        repl_path: str,
        project_dir: str,
        max_repls: int = pool_settings.max_repls,
        max_repl_uses: int = pool_settings.max_repl_uses,
        max_repl_mem: int = pool_settings.max_repl_mem,
        max_wait: float = pool_settings.max_wait,
        init_repls: dict[str, int] | None = None,
    ) -> None:
        self.repl_path = repl_path
        self.project_dir = project_dir
        self.max_repls = max_repls
        self.max_repl_uses = max_repl_uses
        self.max_repl_mem = max_repl_mem
        self.max_wait = max_wait
        self.init_repls = init_repls or {}

        self._lock: asyncio.Lock | None = None
        self._cond: asyncio.Condition | None = None
        self._free: list[Repl] = []
        self._busy: set[Repl] = set()

        logger.info(
            "REPL manager initialized with: MAX_REPLS=%d, MAX_REPL_USES=%d, MAX_REPL_MEM=%d MB",
            max_repls,
            max_repl_uses,
            max_repl_mem,
        )

    def _ensure_lock(self) -> None:
        """Ensure the lock and condition are initialized in an async context."""
        if self._lock is None:
            self._lock = asyncio.Lock()
            self._cond = asyncio.Condition(self._lock)

    async def initialize_repls(self) -> None:
        """Pre-initialize REPLs with common headers."""
        if not self.init_repls:
            return
        if self.max_repls < sum(self.init_repls.values()):
            raise ValueError(
                f"Cannot initialize REPLs: Σ (INIT_REPLS values) = {sum(self.init_repls.values())} > {self.max_repls} = MAX_REPLS"
            )
        initialized_repls: list[Repl] = []
        for header, count in self.init_repls.items():
            for _ in range(count):
                initialized_repls.append(await self.get_repl(header=header))

        async def _prep_and_release(repl: Repl) -> None:
            # All initialized imports should finish in 60 seconds.
            await self.prep(repl, snippet_id="init", timeout=60)
            await self.release_repl(repl)

        await asyncio.gather(*(_prep_and_release(r) for r in initialized_repls))

        logger.info(f"Initialized REPLs with: {json.dumps(self.init_repls, indent=2)}")

    async def get_repl(
        self,
        header: str = "",
        snippet_id: str = "",
        timeout: float | None = None,
        reuse: bool = True,
    ) -> Repl:
        """Get a REPL instance for a given header.

        Three-phase strategy:
        1. Reuse: Check free list for matching header
        2. Create: Spawn new if under capacity
        3. Evict: Destroy oldest free REPL if at capacity
        """
        self._ensure_lock()
        assert self._cond is not None

        if timeout is None:
            timeout = self.max_wait

        deadline = time() + timeout
        repl_to_destroy: Repl | None = None

        while True:
            async with self._cond:
                logger.debug(
                    "Pool status: Free=%d, Busy=%d, Max=%d",
                    len(self._free),
                    len(self._busy),
                    self.max_repls,
                )

                # Phase 1: Reuse existing REPL with matching header
                if reuse:
                    for i, r in enumerate(self._free):
                        if r.header == header:
                            repl = self._free.pop(i)
                            self._busy.add(repl)
                            logger.info(
                                "[%s] Reusing REPL for %s",
                                repl.uuid.hex[:8],
                                snippet_id,
                            )
                            return repl

                # Phase 2: Create new if under capacity
                total = len(self._free) + len(self._busy)
                if total < self.max_repls:
                    break

                # Phase 3: Evict oldest free REPL
                if self._free:
                    oldest = min(self._free, key=lambda r: r.last_check_at)
                    self._free.remove(oldest)
                    repl_to_destroy = oldest
                    break

                # All REPLs busy - wait for one to be released
                remaining = deadline - time()
                if remaining <= 0:
                    raise NoAvailableReplError(f"Timed out after {timeout}s")

                try:
                    logger.info(
                        "Waiting for a REPL to become available (timeout in %.2fs)",
                        remaining,
                    )
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise NoAvailableReplError(
                        f"Timed out after {timeout}s while waiting for a REPL"
                    ) from None

        # Clean up old REPL asynchronously
        if repl_to_destroy is not None:
            asyncio.create_task(close_verbose(repl_to_destroy))

        return await self._start_new(header)

    async def destroy_repl(self, repl: Repl) -> None:
        """Destroy a REPL immediately."""
        self._ensure_lock()
        assert self._cond is not None
        async with self._cond:
            self._busy.discard(repl)
            if repl in self._free:
                self._free.remove(repl)
            asyncio.create_task(close_verbose(repl))
            self._cond.notify(1)

    async def release_repl(self, repl: Repl) -> None:
        """Return a REPL to the pool or destroy if exhausted."""
        self._ensure_lock()
        assert self._cond is not None
        async with self._cond:
            if repl not in self._busy:
                logger.error(
                    "Attempted to release a REPL that is not busy: %s",
                    repl.uuid.hex[:8],
                )
                return

            if repl.exhausted:
                uuid = repl.uuid
                logger.info("REPL %s is exhausted, closing it", uuid.hex[:8])
                self._busy.discard(repl)
                asyncio.create_task(close_verbose(repl))
                self._cond.notify(1)
                return

            self._busy.remove(repl)
            self._free.append(repl)
            repl.last_check_at = datetime.now()
            logger.info("[%s] Released REPL to pool", repl.uuid.hex[:8])
            self._cond.notify(1)

    async def _start_new(self, header: str) -> Repl:
        """Create and start a new REPL."""
        repl = await Repl.create(
            header,
            max_repl_uses=self.max_repl_uses,
            max_repl_mem=self.max_repl_mem,
            repl_path=self.repl_path,
            project_dir=self.project_dir,
        )
        self._busy.add(repl)
        return repl

    async def cleanup(self) -> None:
        """Clean up all REPLs on shutdown."""
        self._ensure_lock()
        assert self._cond is not None
        async with self._cond:
            logger.info("Cleaning up REPL manager...")
            for repl in self._free:
                asyncio.create_task(close_verbose(repl))
            self._free.clear()

            for repl in self._busy:
                asyncio.create_task(close_verbose(repl))
            self._busy.clear()

            logger.info("REPL manager cleaned up!")

    async def prep(
        self, repl: Repl, snippet_id: str, timeout: float
    ) -> ReplResponse | None:
        """Prepare a REPL by starting it and running the header."""
        if repl.is_running:
            return None

        try:
            await repl.start()
        except Exception as e:
            logger.exception("Failed to start REPL: %s", e)
            raise ReplError("Failed to start REPL") from e

        if not is_blank(repl.header):
            try:
                cmd_response = await repl.send_timeout(
                    f"{snippet_id}-header",
                    repl.header,
                    timeout=timeout,
                    is_header=True,
                )
            except asyncio.TimeoutError as e:
                logger.error("Header command timed out")
                raise e
            except Exception as e:
                logger.error("Failed to run header on REPL")
                raise ReplError("Failed to run header on REPL") from e

            if cmd_response.response and cmd_response.response.error:
                logger.error("Header command failed: %s", cmd_response.response.error)
                await self.destroy_repl(repl)

            repl.header_cmd_response = cmd_response
            return cmd_response

        return repl.header_cmd_response

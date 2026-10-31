"""Cancellable POSIX streams for MCP's public stdio transport.

MCP 2.2's default AsyncFile streams wait on blocking worker threads during
cancellation. Explicit streams keep that I/O on the event loop, so shutdown can
reach the application lifespan even while the client holds a pipe open.
"""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
import os
import sys
import threading
from typing import TextIO

import anyio
from anyio.lowlevel import checkpoint
from mcp.server.stdio import stdio_server
from mcp.server.mcpserver import MCPServer


class _NonblockingStdin(anyio.AsyncFile[str]):
    def __init__(self, stream: TextIO) -> None:
        super().__init__(stream)
        self._pending = bytearray()

    async def readline(self) -> str:
        fd = self.wrapped.fileno()
        search_from = 0
        while True:
            await checkpoint()
            newline = self._pending.find(b"\n", search_from)
            if newline >= 0:
                line = self._pending[: newline + 1]
                del self._pending[: newline + 1]
                return line.decode("utf-8", errors="replace")
            search_from = len(self._pending)
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                await anyio.wait_readable(fd)
                continue
            if not chunk:
                tail = self._pending.decode("utf-8", errors="replace")
                self._pending.clear()
                return tail
            self._pending.extend(chunk)


class _NonblockingStdout(anyio.AsyncFile[str]):
    async def write(self, b: str) -> int:
        fd = self.wrapped.fileno()
        encoded = memoryview(b.encode("utf-8"))
        offset = 0
        while offset < len(encoded):
            await checkpoint()
            try:
                offset += os.write(fd, encoded[offset : offset + 65536])
            except BlockingIOError:
                await anyio.wait_writable(fd)
        return len(b)

    async def flush(self) -> None:
        # write() sends bytes directly to the wire; there is no buffered tail.
        await checkpoint()


_stdio_lock = threading.Lock()


@contextmanager
def _cancellable_stdio(
    stdin: TextIO,
    stdout: TextIO,
) -> Iterator[tuple[anyio.AsyncFile[str], anyio.AsyncFile[str]]]:
    # Explicit streams leave descriptor ownership to us. Duplicate the wire,
    # divert tools/children away from it, and unwind everything in reverse order.
    if not _stdio_lock.acquire(blocking=False):
        raise RuntimeError("stdio transport is already in use")
    try:
        with ExitStack() as stack:
            streams = []
            for fd, stream in enumerate((stdin, stdout)):
                wire = stack.enter_context(
                    open(
                        os.dup(stream.buffer.fileno()),
                        "r" if fd == 0 else "w",
                        encoding="utf-8",
                        errors="replace" if fd == 0 else "strict",
                    )
                )
                if stream.buffer.fileno() == fd:
                    stack.callback(os.dup2, wire.fileno(), fd)
                    if fd == 0:
                        with open(os.devnull, "rb") as null:
                            os.dup2(null.fileno(), fd)
                    else:
                        os.dup2(2, fd)
                blocking = os.get_blocking(wire.fileno())
                stack.callback(os.set_blocking, wire.fileno(), blocking)
                os.set_blocking(wire.fileno(), False)
                streams.append(wire)
            yield _NonblockingStdin(streams[0]), _NonblockingStdout(streams[1])
    finally:
        _stdio_lock.release()


class CancellableMCPServer(MCPServer):
    async def run_stdio_async(self) -> None:
        if os.name != "posix":
            await super().run_stdio_async()
            return
        try:
            sys.stdin.buffer.fileno()
            sys.stdout.buffer.fileno()
        except (AttributeError, OSError, ValueError):
            await super().run_stdio_async()
            return
        with _cancellable_stdio(sys.stdin, sys.stdout) as (stdin, stdout):
            # The high-level runner exposes no explicit-stream argument. Keep
            # its low-level bridge here; framing and lifespan stay SDK-owned.
            async with stdio_server(stdin=stdin, stdout=stdout) as (
                reader,
                writer,
            ):
                await self._lowlevel_server.run(
                    reader,
                    writer,
                    self._lowlevel_server.create_initialization_options(),
                )

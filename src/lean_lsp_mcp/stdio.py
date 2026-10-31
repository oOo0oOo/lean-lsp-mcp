"""Cancellable POSIX I/O for the pinned MCP 2.2 stdio transport.

The SDK's threaded I/O cannot be cancelled while a client keeps input open or
stops draining output. Reuse its private descriptor claims/diversion instead of
creating another ownership mechanism. Recheck these helpers and the low-level
run call when updating the pinned SDK; HTTP and non-POSIX transports remain
SDK-owned.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import os
import sys
from typing import Literal, TextIO

import anyio
from anyio.lowlevel import checkpoint
from mcp.server import stdio as sdk_stdio
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


@contextmanager
def _cancellable_stream(
    stream: TextIO, fd: Literal[0, 1]
) -> Iterator[anyio.AsyncFile[str]]:
    wire, restore = (
        sdk_stdio._claim_fd(0, stream, "rb", sdk_stdio._open_stdin_diversion)
        if fd == 0
        else sdk_stdio._claim_fd(1, stream, "wb", sdk_stdio._open_stdout_diversion)
    )
    owns_wire = restore is not None and wire is not getattr(stream, "buffer", None)
    blocking = None
    text_stream = None
    try:
        blocking = os.get_blocking(wire.fileno())
        os.set_blocking(wire.fileno(), False)
        text_stream = sdk_stdio._UnownedTextWrapper(
            wire, encoding="utf-8", errors="replace" if fd == 0 else "strict"
        )
        yield (
            _NonblockingStdin(text_stream)
            if fd == 0
            else _NonblockingStdout(text_stream)
        )
    finally:
        try:
            # Duplicates share their blocking flags with the original wire.
            if blocking is not None:
                os.set_blocking(wire.fileno(), blocking)
        finally:
            if text_stream is not None:
                text_stream.close()
            if restore is not None:
                restore()
            # Failed SDK restoration retains the registered claim and private
            # fd. Never recycle that descriptor or close a borrowed fallback.
            if owns_wire and fd not in sdk_stdio._claims:
                private_fd = wire.fileno()
                wire.close()  # The SDK opens this buffer with closefd=False.
                os.close(private_fd)


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
        with (
            _cancellable_stream(sys.stdin, 0) as stdin,
            _cancellable_stream(sys.stdout, 1) as stdout,
        ):
            # Preserve the SDK's descriptor isolation, framing and lifespan.
            async with sdk_stdio.stdio_server(stdin=stdin, stdout=stdout) as (
                reader,
                writer,
            ):
                await self._lowlevel_server.run(
                    reader,
                    writer,
                    self._lowlevel_server.create_initialization_options(),
                )

import asyncio
from contextlib import contextmanager
import io
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import AsyncMock

import pytest

from lean_lsp_mcp import stdio


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX nonblocking I/O")


@pytest.fixture
def waiting_for_input(monkeypatch):
    waiting = asyncio.Event()
    original = stdio.anyio.wait_readable

    async def wait_readable(fd):
        waiting.set()
        await original(fd)

    monkeypatch.setattr(stdio.anyio, "wait_readable", wait_readable)
    return waiting


@contextmanager
def pipe_reader():
    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    with os.fdopen(reader, "rb", closefd=False) as wire:
        text = io.TextIOWrapper(wire, encoding="utf-8")
        try:
            yield stdio._NonblockingStdin(text), writer
        finally:
            text.detach()
            os.close(reader)
            os.close(writer)


@contextmanager
def pipe_writer():
    reader, writer = os.pipe()
    os.set_blocking(writer, False)
    with os.fdopen(writer, "wb", closefd=False) as wire:
        text = io.TextIOWrapper(wire, encoding="utf-8")
        try:
            yield reader, stdio._NonblockingStdout(text)
        finally:
            text.detach()
            os.close(reader)
            os.close(writer)


@pytest.mark.asyncio
async def test_fragmented_utf8_crlf_and_buffered_lines(waiting_for_input):
    with pipe_reader() as (reader, writer):
        pending = asyncio.create_task(reader.readline())
        try:
            os.write(writer, b'"\xe2')
            await asyncio.wait_for(waiting_for_input.wait(), 1)
            assert not pending.done()
            os.write(writer, b'\x88\x80"\r\nsecond\n')
            assert await asyncio.wait_for(pending, 1) == '"∀"\r\n'
            assert await reader.readline() == "second\n"
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [b"", b"partial \xe2"])
async def test_cancel_read_with_writer_still_open(prefix, waiting_for_input):
    with pipe_reader() as (reader, writer):
        if prefix:
            os.write(writer, prefix)
        pending = asyncio.create_task(reader.readline())
        try:
            await asyncio.wait_for(waiting_for_input.wait(), 1)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 1)
            # Cancellation does not discard bytes already consumed from wire.
            suffix = b"\x88\x80\n" if prefix else b"next\n"
            os.write(writer, suffix)
            expected = "partial ∀\n" if prefix else "next\n"
            assert await asyncio.wait_for(reader.readline(), 1) == expected
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_regular_file_eof_tail_and_invalid_utf8(tmp_path, monkeypatch):
    path = tmp_path / "messages"
    path.write_bytes(b"first\nlast\xff")
    monkeypatch.setattr(
        stdio.anyio, "wait_readable", AsyncMock(side_effect=AssertionError)
    )
    with path.open("r") as stream:
        reader = stdio._NonblockingStdin(stream)
        assert await reader.readline() == "first\n"
        assert await reader.readline() == "last�"
        assert await reader.readline() == ""
    with open(os.devnull) as stream:
        assert await stdio._NonblockingStdin(stream).readline() == ""


@pytest.mark.asyncio
async def test_output_partial_writes_preserve_utf8_and_character_count(monkeypatch):
    original_write = os.write
    original_wait = stdio.anyio.wait_writable
    waits = []
    blocked_once = False

    async def wait_writable(fd):
        waits.append(fd)
        await original_wait(fd)

    def short_write(fd, data):
        nonlocal blocked_once
        if not blocked_once:
            blocked_once = True
            raise BlockingIOError
        return original_write(fd, data[:2])

    monkeypatch.setattr(stdio.os, "write", short_write)
    monkeypatch.setattr(stdio.anyio, "wait_writable", wait_writable)
    data = '"∀𝒜"\n'
    with pipe_writer() as (reader, writer):
        assert await writer.write(data) == len(data)
        await writer.flush()
        assert os.read(reader, 1024) == data.encode("utf-8")
    assert len(waits) == 1


@pytest.mark.asyncio
async def test_cancel_write_with_output_pipe_full(monkeypatch):
    waiting = asyncio.Event()
    original_wait = stdio.anyio.wait_writable

    async def wait_writable(fd):
        waiting.set()
        await original_wait(fd)

    monkeypatch.setattr(stdio.anyio, "wait_writable", wait_writable)
    with pipe_writer() as (reader, writer):
        pending = asyncio.create_task(writer.write("x" * (1024 * 1024)))
        try:
            await asyncio.wait_for(waiting.wait(), 1)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 1)
            # A partial frame can be abandoned at shutdown, but no writer
            # worker survives cancellation and continues emitting its tail.
            os.set_blocking(reader, False)
            while True:
                try:
                    assert os.read(reader, 65536)
                except BlockingIOError:
                    break
            assert await writer.write("next\n") == 5
            await writer.flush()
            assert os.read(reader, 1024) == b"next\n"
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_output_regular_file_and_broken_pipe(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stdio.anyio, "wait_writable", AsyncMock(side_effect=AssertionError)
    )
    path = tmp_path / "output"
    with path.open("w") as stream:
        writer = stdio._NonblockingStdout(stream)
        assert await writer.write("∀\n") == 2
        await writer.flush()
    assert path.read_bytes() == "∀\n".encode("utf-8")
    reader, writer_fd = os.pipe()
    os.close(reader)
    with os.fdopen(writer_fd, "w") as stream:
        writer = stdio._NonblockingStdout(stream)
        with pytest.raises(BrokenPipeError):
            await writer.write("message\n")


@pytest.mark.parametrize("fail", [False, True])
def test_stream_flags_and_descriptor_ownership(monkeypatch, tmp_path, fail):
    input_path = tmp_path / "input"
    input_path.write_text("message\n")
    original_dup = os.dup
    private_fds = []

    def duplicate(fd):
        private_fd = original_dup(fd)
        private_fds.append(private_fd)
        return private_fd

    monkeypatch.setattr(stdio.os, "dup", duplicate)
    with input_path.open() as stdin, (tmp_path / "output").open("w") as stdout:
        try:
            with stdio._cancellable_stdio(stdin, stdout):
                assert not os.get_blocking(stdin.fileno())
                assert not os.get_blocking(stdout.fileno())
                if fail:
                    raise RuntimeError("interrupted")
        except RuntimeError:
            assert fail
        assert os.get_blocking(stdin.fileno())
        assert os.get_blocking(stdout.fileno())
        for fd in private_fds:
            with pytest.raises(OSError):
                os.fstat(fd)
        assert stdin.readline() == "message\n"
        stdout.write("still open\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", ["stdin", "stdout"])
async def test_no_descriptor_uses_sdk_runner(monkeypatch, stream):
    monkeypatch.setattr(stdio.sys, stream, io.StringIO())
    original = AsyncMock()
    monkeypatch.setattr(stdio.MCPServer, "run_stdio_async", original)
    await stdio.CancellableMCPServer(name="test").run_stdio_async()
    original.assert_awaited_once()


def test_real_stdio_isolation_exclusion_and_release():
    script = textwrap.dedent(r"""
        import asyncio, os, sys
        from lean_lsp_mcp.stdio import _cancellable_stdio
        async def check():
            with _cancellable_stdio(sys.stdin, sys.stdout) as (reader, writer):
                private_fds = [reader.wrapped.fileno(), writer.wrapped.fileno()]
                assert os.read(0, 1) == b''
                os.write(1, b'diverted\n')
                try:
                    with _cancellable_stdio(sys.stdin, sys.stdout):
                        raise AssertionError('second transport accepted')
                except RuntimeError:
                    pass
                assert await writer.write('∀ ' + await reader.readline()) == 8
                await writer.flush()
            assert os.get_blocking(0)
            assert os.get_blocking(1)
            for private_fd in private_fds:
                try:
                    os.fstat(private_fd)
                except OSError:
                    pass
                else:
                    raise AssertionError('private fd leaked')
            print('restored')
        asyncio.run(check())
    """)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        input="hello\n",
        text=True,
        capture_output=True,
        timeout=5,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "∀ hello\nrestored\n"
    assert result.stderr == "diverted\n"

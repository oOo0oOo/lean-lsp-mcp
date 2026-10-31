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
        text = stdio.sdk_stdio._UnownedTextWrapper(wire, encoding="utf-8")
        try:
            yield stdio._NonblockingStdin(text), writer
        finally:
            text.close()
            os.close(reader)
            os.close(writer)


@contextmanager
def pipe_writer():
    reader, writer = os.pipe()
    os.set_blocking(writer, False)
    with os.fdopen(writer, "wb", closefd=False) as wire:
        text = stdio.sdk_stdio._UnownedTextWrapper(wire, encoding="utf-8")
        try:
            yield reader, stdio._NonblockingStdout(text)
        finally:
            text.close()
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


@pytest.mark.parametrize("owned,released", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("fd", [0, 1])
@pytest.mark.parametrize("interrupted", [False, True])
def test_stream_flags_and_descriptor_ownership(
    monkeypatch, owned, released, fd, interrupted
):
    reader, writer = os.pipe()
    original_fd = reader if fd == 0 else writer
    mode = "rb" if fd == 0 else "wb"
    private_fd = os.dup(original_fd) if owned else original_fd
    wire = os.fdopen(private_fd, mode, closefd=False)
    stream = io.TextIOWrapper(os.fdopen(original_fd, mode, closefd=False))
    if not owned:
        wire.close()
        wire = stream.buffer
    claims = {fd: object()}

    def restore():
        if released:
            claims.clear()

    monkeypatch.setattr(stdio.sdk_stdio, "_claim_fd", lambda *args: (wire, restore))
    monkeypatch.setattr(stdio.sdk_stdio, "_claims", claims)
    try:
        try:
            with stdio._cancellable_stream(stream, fd):
                assert not os.get_blocking(original_fd)
                if interrupted:
                    raise RuntimeError("interrupted")
        except RuntimeError:
            assert interrupted
        assert os.get_blocking(original_fd)
        if owned and released:
            with pytest.raises(OSError):
                os.fstat(private_fd)
        else:
            os.fstat(private_fd)
        assert (fd in claims) is not released
    finally:
        stream.close()
        if owned and not released:
            wire.close()
            os.close(private_fd)
        os.close(reader)
        os.close(writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", ["stdin", "stdout"])
async def test_no_descriptor_uses_sdk_runner(monkeypatch, stream):
    monkeypatch.setattr(stdio.sys, stream, io.StringIO())
    original = AsyncMock()
    monkeypatch.setattr(stdio.MCPServer, "run_stdio_async", original)
    await stdio.CancellableMCPServer(name="test").run_stdio_async()
    original.assert_awaited_once()


def test_real_sdk_claim_exclusion_and_release():
    script = textwrap.dedent(r"""
        import asyncio, os, sys
        from lean_lsp_mcp.stdio import _cancellable_stream
        async def check():
            with (_cancellable_stream(sys.stdin, 0) as reader,
                  _cancellable_stream(sys.stdout, 1) as writer):
                private_fds = [reader.wrapped.fileno(), writer.wrapped.fileno()]
                assert os.read(0, 1) == b''
                os.write(1, b'diverted\n')
                for fd, stream in enumerate([sys.stdin, sys.stdout]):
                    try:
                        with _cancellable_stream(stream, fd):
                            raise AssertionError('second claim accepted')
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


def _run_claim_probe(body: str) -> subprocess.CompletedProcess[str]:
    # Claims deliberately retained after a restore failure must not contaminate
    # this pytest process or its stdout capture.
    preamble = textwrap.dedent("""
        import asyncio, io, os, sys
        from contextlib import contextmanager
        import anyio
        from mcp.server import stdio as sdk
        from lean_lsp_mcp.stdio import _cancellable_stream

        @contextmanager
        def claimed_stdio():
            with (_cancellable_stream(sys.stdin, 0) as reader,
                  _cancellable_stream(sys.stdout, 1) as writer):
                yield reader, writer
    """)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-B", "-c", preamble + textwrap.dedent(body)],
        input="hello\n",
        text=True,
        capture_output=True,
        timeout=5,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.mark.parametrize("fd", [0, 1])
def test_restore_syscall_failure_retains_wire_and_refuses_successors(fd):
    result = _run_claim_probe(f"""
        import errno
        report = os.dup(1)
        original_dup2 = os.dup2
        failed_wire = None
        failures = 0
        def fail_restore(source, target, *args, **kwargs):
            global failures
            if source == failed_wire and target == {fd}:
                failures += 1
                raise OSError(errno.EIO, 'injected restore failure')
            return original_dup2(source, target, *args, **kwargs)
        os.dup2 = fail_restore
        try:
            with claimed_stdio() as streams:
                failed_wire = streams[{fd}].wrapped.fileno()
        finally:
            os.dup2 = original_dup2
        assert failures == 1
        assert set(sdk._claims) == {{{fd}}}
        os.fstat(failed_wire)
        assert os.get_blocking(failed_wire)
        try:
            with claimed_stdio():
                raise AssertionError('adapter accepted diverted descriptors')
        except RuntimeError:
            pass
        async def sdk_successor():
            try:
                async with sdk.stdio_server():
                    raise AssertionError('SDK accepted diverted descriptors')
            except RuntimeError:
                pass
        asyncio.run(sdk_successor())
        assert set(sdk._claims) == {{{fd}}}
        os.fstat(failed_wire)
        os.write(report, b'claim and wire retained\\n')
    """)
    assert result.stdout == "claim and wire retained\n"


@pytest.mark.parametrize("first", ["sdk", "adapter"])
def test_mixed_sdk_and_adapter_transports_share_claim_exclusion(first):
    result = _run_claim_probe(f"""
        class Finished(Exception):
            pass
        def only_finished(exc):
            if isinstance(exc, Finished):
                return True
            children = getattr(exc, 'exceptions', ())
            return bool(children) and all(only_finished(e) for e in children)
        async def check():
            if {first!r} == 'sdk':
                try:
                    async with sdk.stdio_server():
                        assert set(sdk._claims) == {{0, 1}}
                        try:
                            with claimed_stdio():
                                raise AssertionError('adapter accepted SDK claims')
                        except RuntimeError:
                            pass
                        # Cancel the SDK's readers/writers without sending any
                        # messages; stdin is already closed by subprocess.run.
                        raise Finished
                except BaseException as exc:
                    if not only_finished(exc):
                        raise
            else:
                with claimed_stdio():
                    assert set(sdk._claims) == {{0, 1}}
                    try:
                        async with sdk.stdio_server():
                            raise AssertionError('SDK accepted adapter claims')
                    except RuntimeError:
                        pass
            assert not sdk._claims
            print('mixed transport refused')
        asyncio.run(check())
    """)
    assert result.stdout == "mixed transport refused\n"


def test_closed_stderr_keeps_protocol_descriptors_out_of_standard_range():
    result = _run_claim_probe(r"""
        async def check():
            # Close it after event-loop setup so no intervening allocator can
            # reuse fd 2 before the transport claims its descriptors.
            os.close(2)
            with claimed_stdio() as (reader, writer):
                private_fds = [reader.wrapped.fileno(), writer.wrapped.fileno()]
                assert all(fd > 2 for fd in private_fds)
                assert len(set(private_fds)) == 2
                assert os.read(0, 1) == b''
                os.write(1, b'stray output must go to the null device\n')
                assert await reader.readline() == 'hello\n'
                await writer.write('wire output\n')
                await writer.flush()
            for fd in private_fds:
                try:
                    os.fstat(fd)
                except OSError:
                    pass
                else:
                    raise AssertionError('private fd leaked')
            try:
                os.fstat(2)
            except OSError:
                pass
            else:
                raise AssertionError('stderr unexpectedly reopened')
            print('restored')
        asyncio.run(check())
    """)
    assert result.stdout == "wire output\nrestored\n"
    assert result.stderr == ""


def test_pinned_sdk_still_has_the_private_stdio_helpers_we_reuse():
    """stdio.py reuses private helpers of the pinned MCP SDK. An SDK update that
    renames or removes them must fail here, not in a client's session."""
    from mcp.server import stdio as sdk_stdio

    for name in (
        "_claim_fd",
        "_claims",
        "_open_stdin_diversion",
        "_open_stdout_diversion",
        "_UnownedTextWrapper",
    ):
        assert hasattr(sdk_stdio, name), (
            f"mcp.server.stdio.{name} is gone; recheck stdio.py"
        )

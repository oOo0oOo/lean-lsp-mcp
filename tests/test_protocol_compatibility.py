"""Wire compatibility checks that require neither a Lean installation nor internet."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx2
from jsonschema import Draft202012Validator
from mcp import Client, ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import LATEST_PROTOCOL_VERSION, RequestParamsMeta
import pytest


@pytest.fixture
def protocol_env(tmp_path: Path, repo_root: Path) -> dict[str, str]:
    # HTTP requires a project root, but listing tools never starts lake serve.
    (tmp_path / "lean-toolchain").write_text("leanprover/lean4:v4.28.0\n")
    (tmp_path / "lakefile.toml").write_text('name = "protocol_fixture"\n')
    # Keep inherited user configuration and secrets out of the fixture process.
    env = {
        k: os.environ[k]
        for k in ("PATH", "HOME", "LANG", "TMPDIR", "SystemRoot")
        if k in os.environ
    }
    env.update(
        PYTHONPATH=str(repo_root / "src") + os.pathsep + str(repo_root),
        LEAN_PROJECT_PATH=str(tmp_path),
        LEAN_LOG_LEVEL="ERROR",
        LEAN_LSP_TEST_MODE="1",
        LEAN_MCP_DISABLED_TOOLS="lean_run_code",
        LEAN_MCP_INSTRUCTIONS="Protocol compatibility fixture instructions",
        LEAN_MCP_TOOL_DESCRIPTIONS='{"lean_goal":"Protocol fixture goal description"}',
    )
    return env


@asynccontextmanager
async def wire_transport(transport: str, env: dict[str, str], url: str | None = None):
    if transport == "stdio":
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "lean_lsp_mcp", "--transport", "stdio"],
            env=env,
        )
        async with stdio_client(params) as streams:
            yield streams
    elif transport == "sse":
        async with sse_client(url) as streams:
            yield streams
    else:
        async with streamable_http_client(url) as streams:
            yield streams[:2]


@asynccontextmanager
async def running_http(
    env: dict[str, str], transport: str = "streamable-http", script: str | None = None
):
    # Select a loopback port; readiness polling is bounded and checks process exit.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "lean_lsp_mcp",
        "--transport",
        transport,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if script is not None:
        command = [sys.executable, "-c", script, str(port)]
    with subprocess.Popen(
        command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    ) as process:
        url = f"http://127.0.0.1:{port}/{'sse' if transport == 'sse' else 'mcp'}"
        try:
            deadline = time.monotonic() + 15
            async with httpx2.AsyncClient(timeout=0.5) as http:
                while True:
                    if process.poll() is not None:
                        raise AssertionError(process.stderr.read().decode())
                    try:
                        response = await http.post(url, json={})
                        if response.status_code:
                            break
                    except httpx2.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        raise AssertionError("MCP HTTP server did not become ready")
                    await asyncio.sleep(0.05)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def assert_local_refs_resolve(schema: dict) -> None:
    def walk(value):
        if isinstance(value, dict):
            ref = value.get("$ref")
            if ref is not None:
                assert ref.startswith("#/"), ref
                target = schema
                for part in ref[2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)


async def contract(transport, env, url=None, *, modern):
    async with wire_transport(transport, env, url) as streams:
        if modern:
            # The modern client gets exactly the same real transport streams.
            @asynccontextmanager
            async def connected():
                yield streams

            async with Client(connected(), cache=None) as client:
                assert client.protocol_version == LATEST_PROTOCOL_VERSION
                assert client.instructions == env["LEAN_MCP_INSTRUCTIONS"]
                tools = await client.list_tools()
                error = await client.call_tool("lean_goal", {})
        else:
            async with ClientSession(*streams) as client:
                initialized = await client.initialize()
                assert initialized.protocol_version != LATEST_PROTOCOL_VERSION
                assert initialized.instructions == env["LEAN_MCP_INSTRUCTIONS"]
                tools = await client.list_tools()
                error = await client.call_tool("lean_goal", {})
    names = [tool.name for tool in tools.tools]
    assert names == sorted(names)
    assert "lean_run_code" not in names
    assert "lean_goal" in names
    assert (
        next(t for t in tools.tools if t.name == "lean_goal").description
        == "Protocol fixture goal description"
    )
    for tool in tools.tools:
        Draft202012Validator.check_schema(tool.input_schema)
        assert_local_refs_resolve(tool.input_schema)
        if tool.output_schema is not None:
            Draft202012Validator.check_schema(tool.output_schema)
            assert_local_refs_resolve(tool.output_schema)
    assert error.is_error
    assert error.content
    if modern:
        assert tools.ttl_ms == 60000
        assert tools.cache_scope == "private"
    # Modern response envelopes carry cache hints and server identity; compare
    # the actual tool contract and result payload across protocol versions.
    return [tool.model_dump(mode="json", exclude_none=True) for tool in tools.tools], {
        "content": [
            block.model_dump(mode="json", exclude_none=True) for block in error.content
        ],
        "structured_content": error.structured_content,
        "is_error": error.is_error,
    }


@pytest.mark.timeout(60)
async def test_stdio_and_http_modern_and_legacy_contracts(protocol_env):
    baseline = await contract("stdio", protocol_env, modern=False)
    assert await contract("stdio", protocol_env, modern=True) == baseline
    async with running_http(protocol_env) as url:
        assert (
            await contract("streamable-http", protocol_env, url, modern=False)
            == baseline
        )
        assert (
            await contract("streamable-http", protocol_env, url, modern=True)
            == baseline
        )


@pytest.mark.timeout(30)
async def test_legacy_sse_contract(protocol_env):
    baseline = await contract("stdio", protocol_env, modern=False)
    async with running_http(protocol_env, "sse") as url:
        assert await contract("sse", protocol_env, url, modern=False) == baseline


@pytest.mark.timeout(30)
@pytest.mark.parametrize("transport", ["streamable-http", "sse"])
async def test_http_bearer_token(protocol_env, transport):
    protocol_env["LEAN_LSP_MCP_TOKEN"] = "protocol-test-token"
    async with running_http(protocol_env, transport) as url:
        async with httpx2.AsyncClient(timeout=5) as http:
            for headers in ({}, {"Authorization": "Bearer wrong-token"}):
                if transport == "sse":
                    response = await http.get(url, headers=headers)
                else:
                    response = await http.post(url, json={}, headers=headers)
                assert response.status_code == 401
        if transport == "streamable-http":
            async with httpx2.AsyncClient(
                headers={"Authorization": "Bearer protocol-test-token"}
            ) as http:
                async with streamable_http_client(url, http_client=http) as streams:
                    async with ClientSession(*streams[:2]) as client:
                        await client.initialize()
                        assert (await client.list_tools()).tools
        else:
            async with sse_client(
                url, headers={"Authorization": "Bearer protocol-test-token"}
            ) as streams:
                async with ClientSession(*streams) as client:
                    await client.initialize()
                    assert (await client.list_tools()).tools


# Run the production factory and lifespan, replacing only the Lean operation
# with an observable deterministic tool. This exercises actual wire dispatch.
PROBE_SERVER = r"""
import asyncio
import os
from pathlib import Path
import sys
import uvicorn
from mcp.server.mcpserver import Context
from lean_lsp_mcp.server import create_server
from lean_lsp_mcp.utils import LeanToolError

server = create_server()
@server.tool()
async def protocol_probe(ctx: Context, wait: bool = False) -> dict[str, str]:
    await ctx.report_progress(1, 2, "started")
    if wait:
        try:
            await asyncio.Event().wait()
        finally:
            Path(os.environ["PROTOCOL_CANCEL_FILE"]).write_text("cancelled")
    await ctx.report_progress(2, 2, "finished")
    return {"status": "complete"}

@server.tool()
async def protocol_log(ctx: Context) -> str:
    await ctx.debug("debug fixture", logger_name="protocol-fixture")
    await ctx.info("info fixture", logger_name="protocol-fixture")
    await ctx.warning("warning fixture", logger_name="protocol-fixture")
    return "logged"

@server.tool()
async def protocol_expected_error() -> str:
    raise LeanToolError("Actionable fixture error: check the Lean project path")

@server.tool()
async def protocol_unexpected_error() -> str:
    raise RuntimeError("Private fixture implementation detail must not reach clients")

if sys.argv[1] == "stdio":
    server.run(transport="stdio")
else:
    app = server.streamable_http_app(session_idle_timeout=float(os.environ.get("PROTOCOL_SESSION_TTL", "30")))
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="error")
"""


@asynccontextmanager
async def probe_transport(transport, env):
    if transport == "stdio":
        params = StdioServerParameters(
            command=sys.executable, args=["-c", PROBE_SERVER, "stdio"], env=env
        )
        async with stdio_client(params) as streams:
            yield streams
    else:
        env["LEAN_LSP_MCP_ACTIVE_TRANSPORT"] = transport
        async with running_http(env, script=PROBE_SERVER) as url:
            async with streamable_http_client(url) as streams:
                yield streams[:2]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("transport", ["stdio", "streamable-http"])
@pytest.mark.parametrize("modern", [False, True])
async def test_progress_cancellation_and_next_request(
    protocol_env, tmp_path, transport, modern
):
    cancel_file = tmp_path / "cancel-observed"
    protocol_env["PROTOCOL_CANCEL_FILE"] = str(cancel_file)
    progress = []
    started = asyncio.Event()

    async def on_progress(current, total, message):
        progress.append((current, total, message))
        started.set()

    async with probe_transport(transport, protocol_env) as streams:

        @asynccontextmanager
        async def connected():
            yield streams

        session = Client(connected(), cache=None) if modern else ClientSession(*streams)
        async with session as client:
            if not modern:
                await client.initialize()
            result = await client.call_tool(
                "protocol_probe", {}, progress_callback=on_progress
            )
            assert not result.is_error
            assert result.structured_content == {"status": "complete"}
            probe = next(
                tool
                for tool in (await client.list_tools()).tools
                if tool.name == "protocol_probe"
            )
            assert probe.output_schema is not None
            Draft202012Validator.check_schema(probe.output_schema)
            Draft202012Validator(probe.output_schema).validate(
                result.structured_content
            )
            text_blocks = [
                block.text for block in result.content if block.type == "text"
            ]
            assert len(text_blocks) == 1
            assert json.loads(text_blocks[0]) == result.structured_content
            assert progress == [(1, 2, "started"), (2, 2, "finished")]
            started.clear()
            task = asyncio.create_task(
                client.call_tool(
                    "protocol_probe", {"wait": True}, progress_callback=on_progress
                )
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            deadline = time.monotonic() + 5
            while not cancel_file.exists():
                assert time.monotonic() < deadline, (
                    "Server did not cancel the running tool"
                )
                await asyncio.sleep(0.02)
            # Cancellation must leave the connection usable.
            assert (await client.list_tools()).tools
            assert not (await client.call_tool("protocol_probe", {})).is_error


@pytest.mark.timeout(30)
async def test_http_idle_session_expiry(protocol_env):
    protocol_env["LEAN_LSP_MCP_ACTIVE_TRANSPORT"] = "streamable-http"
    protocol_env["PROTOCOL_SESSION_TTL"] = "0.2"
    async with running_http(protocol_env, script=PROBE_SERVER) as url:
        headers = {
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
        }
        async with httpx2.AsyncClient(timeout=5) as http:
            response = await http.post(
                url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "expiry-fixture", "version": "1"},
                    },
                },
            )
            assert response.status_code == 200
            session_id = response.headers["mcp-session-id"]
            headers["Mcp-Session-Id"] = session_id
            response = await http.post(
                url,
                headers=headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            assert response.status_code == 202
            # No GET channel remains open, and no request refreshes the deadline.
            await asyncio.sleep(0.5)
            response = await http.post(
                url,
                headers=headers,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            assert response.status_code == 404


@pytest.mark.timeout(30)
@pytest.mark.parametrize("transport", ["stdio", "streamable-http"])
async def test_modern_logging_requires_per_request_opt_in(protocol_env, transport):
    messages = []
    warning_received = asyncio.Event()

    async def on_log(params):
        messages.append((params.level, params.logger, params.data))
        if params.level == "warning":
            warning_received.set()

    async with probe_transport(transport, protocol_env) as streams:

        @asynccontextmanager
        async def connected():
            yield streams

        async with Client(connected(), cache=None, logging_callback=on_log) as client:
            assert client.protocol_version == LATEST_PROTOCOL_VERSION
            # Installing a callback alone does not opt a modern request into logs.
            assert not (await client.call_tool("protocol_log", {})).is_error
            assert messages == []
            await client.call_tool(
                "protocol_log",
                {},
                meta=RequestParamsMeta(
                    **{
                        "io.modelcontextprotocol/logLevel": "info",
                    }
                ),
            )
            await asyncio.wait_for(warning_received.wait(), timeout=5)
            assert messages == [
                ("info", "protocol-fixture", "info fixture"),
                ("warning", "protocol-fixture", "warning fixture"),
            ]
            # The opt-in belongs to that request, not the entire connection.
            await client.call_tool("protocol_log", {})
            assert len(messages) == 2
            warning_received.clear()
            await client.call_tool(
                "protocol_log",
                {},
                meta=RequestParamsMeta(
                    **{
                        "io.modelcontextprotocol/logLevel": "warning",
                    }
                ),
            )
            await asyncio.wait_for(warning_received.wait(), timeout=5)
            assert messages[2:] == [("warning", "protocol-fixture", "warning fixture")]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("transport", ["stdio", "streamable-http"])
@pytest.mark.parametrize("modern", [False, True])
async def test_expected_tool_error_visible_unexpected_error_redacted(
    protocol_env, transport, modern
):
    async with probe_transport(transport, protocol_env) as streams:

        @asynccontextmanager
        async def connected():
            yield streams

        session = Client(connected(), cache=None) if modern else ClientSession(*streams)
        async with session as client:
            if not modern:
                await client.initialize()
            expected = await client.call_tool("protocol_expected_error", {})
            assert expected.is_error
            expected_text = "\n".join(
                block.text for block in expected.content if block.type == "text"
            )
            assert (
                "Actionable fixture error: check the Lean project path" in expected_text
            )
            unexpected = await client.call_tool("protocol_unexpected_error", {})
            assert unexpected.is_error
            assert unexpected.content
            wire_payload = json.dumps(unexpected.model_dump(mode="json"))
            assert "Private fixture implementation detail" not in wire_payload
            assert "RuntimeError" not in wire_payload
            assert not (await client.call_tool("protocol_probe", {})).is_error

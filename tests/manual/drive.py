"""Manual MCP driver: spawn lean-lsp-mcp over stdio and time every tool call.

Usage:
    .venv/bin/python tests/manual/drive.py tests/manual/calls_smoke.json

Call-file format (JSON list, executed in order):
    {"tool": "lean_goal", "args": {...}, "label": "...", "show": 400, "timeout": 180}
    {"exec": "shell command"}            # run between tool calls (e.g. edit a file)
    {"gather": [ {tool...}, {tool...} ]} # fire concurrently (parallelism check)

The server runs against tests/test_project by default (override PROJ env).
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]
PROJ = os.environ.get("PROJ", str(REPO / "tests/test_project"))
SRV = os.environ.get("SRV", str(REPO / ".venv/bin/lean-lsp-mcp"))

CALLS = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else []


async def main():
    params = StdioServerParameters(
        command=SRV,
        env={**os.environ, "LEAN_PROJECT_PATH": PROJ},
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as sess:
            t0 = time.time()
            await sess.initialize()
            print(f"[{time.time() - t0:7.2f}s] initialize")
            for call in CALLS:
                if "exec" in call:
                    r2 = subprocess.run(
                        call["exec"], shell=True, capture_output=True, text=True
                    )
                    print(
                        f"[  exec  ] {call.get('label', call['exec'])} "
                        f"rc={r2.returncode} {r2.stdout.strip()[:200]}"
                    )
                    continue
                if "gather" in call:

                    async def one(c):
                        t = time.time()
                        res = await sess.call_tool(c["tool"], c.get("args", {}))
                        return c.get("label", c["tool"]), time.time() - t, res.isError

                    t = time.time()
                    results = await asyncio.gather(*(one(c) for c in call["gather"]))
                    print(f"[{time.time() - t:7.2f}s] GATHER total")
                    for lbl, dt, err in results:
                        print(f"    [{dt:7.2f}s] {lbl} isError={err}")
                    continue
                name, args = call["tool"], call.get("args", {})
                label = call.get("label", name)
                t = time.time()
                try:
                    res = await asyncio.wait_for(
                        sess.call_tool(name, args), timeout=call.get("timeout", 180)
                    )
                    dt = time.time() - t
                    out = "\n".join(c.text for c in res.content if hasattr(c, "text"))
                    print(f"[{dt:7.2f}s] {label} isError={res.isError} len={len(out)}")
                    maxlen = call.get("show", 400)
                    print("    " + out[:maxlen].replace("\n", "\n    "))
                    if len(out) > maxlen:
                        print(f"    ...[{len(out)} chars total]")
                except asyncio.TimeoutError:
                    print(f"[TIMEOUT>{call.get('timeout', 180)}s] {label}")


asyncio.run(main())

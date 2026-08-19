"""Transport-independent helpers shared by MCP tool implementations."""

from __future__ import annotations

import asyncio
import functools
import ssl
import time
import urllib.request
from collections.abc import Callable
from typing import Any

import certifi
import orjson

from lean_lsp_mcp import config
from lean_lsp_mcp.utils import LeanToolError


async def safe_report_progress(
    ctx: Any, *, progress: int, total: int, message: str
) -> None:
    try:
        await ctx.report_progress(progress=progress, total=total, message=message)
    except Exception:
        return


async def urlopen_json(req: urllib.request.Request, timeout: float) -> Any:
    """Run a JSON HTTP request in a worker thread."""
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    def request() -> Any:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as response:
            return orjson.loads(response.read())

    return await asyncio.to_thread(request)


def custom_backend(env_var: str, default_url: str) -> bool:
    """Return whether a tool uses a self-hosted rather than public backend."""
    return config.is_custom_backend(env_var, default_url)


def rate_limited(
    category: str,
    max_requests: int,
    per_seconds: int,
    bypass: Callable[[], bool] | None = None,
):
    def decorator(func):
        def apply(args, kwargs) -> str | None:
            if bypass is not None and bypass():
                return None
            ctx = kwargs.get("ctx")
            if ctx is None:
                if not args:
                    raise KeyError(
                        "rate_limited wrapper requires ctx as a keyword argument or the first positional argument"
                    )
                ctx = args[0]
            rate_limits = ctx.request_context.lifespan_context.rate_limit
            now = int(time.time())
            rate_limits[category] = [
                timestamp
                for timestamp in rate_limits[category]
                if timestamp > now - per_seconds
            ]
            if len(rate_limits[category]) >= max_requests:
                return (
                    f"Tool limit exceeded: {max_requests} requests per "
                    f"{per_seconds} s. Try again later."
                )
            rate_limits[category].append(now)
            return None

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if message := apply(args, kwargs):
                    raise LeanToolError(message)
                return await func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if message := apply(args, kwargs):
                    raise LeanToolError(message)
                return func(*args, **kwargs)

        wrapper.__doc__ = (
            f"Limit: {max_requests}req/{per_seconds}s. {wrapper.__doc__ or ''}"
        )
        return wrapper

    return decorator

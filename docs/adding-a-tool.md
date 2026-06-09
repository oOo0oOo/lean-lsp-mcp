# How to add a new MCP tool

A tool in this repository is a thin `@mcp.tool` wrapper in `src/lean_lsp_mcp/server.py`. Everything else (return shape, helpers, tests, docs) lives in sibling files. Adding a new tool typically touches **5 places**, in this order.

## Checklist

1. **`src/lean_lsp_mcp/models.py`** — add the Pydantic return model.
2. **`src/lean_lsp_mcp/<your_helper>.py`** — put non-trivial logic here (optional).
3. **`src/lean_lsp_mcp/server.py`** — register the tool with `@mcp.tool(...)`.
4. **`tests/test_<area>.py`** — add an end-to-end test that invokes the tool through `MCPClient`.
5. **`docs/tools.md`** — add a short user-facing description.

If the tool needs new lifespan-scoped state (a long-running subprocess for instance), also touch `AppContext` and `app_lifespan` in `server.py`.

---

## 1. Define the return model — `models.py`

Tools return a Pydantic model (not a dict, not a raw string). One model per tool, named `<Thing>Result` for single objects or `<Thing>Results` (plural) for list wrappers.

```python
# in src/lean_lsp_mcp/models.py
class MyToolResult(BaseModel):
    name: str = Field(description="...")
    line: int = Field(description="1-indexed line")
    matches: List[str] = Field(default_factory=list, description="...")
```

Conventions used elsewhere in this file:

- All fields get a `description=` (it shows up in the MCP schema).
- 1-indexed line/column at the boundary; convert to 0-indexed only inside the implementation.
- For "list of X" outputs, wrap in a `XResults` model with an `items: List[X]` field — bare `List[...]` returns are avoided.
- Use `Optional[...]` + `None` default, not `| None` defaults inline.
- Use `Enum` subclasses for closed sets of strings (see `DiagnosticSeverity`).

Then export the new model via the import block at the top of `server.py` (the big `from lean_lsp_mcp.models import (...)`).

## 2. Put the heavy logic in a helper module

If the body is more than ~30 lines or pulls in subprocesses, parsers, or network code, give it its own module rather than bloating `server.py`. Pick the convention that matches existing tools:

| Domain                    | File                | Used by                              |
| ------------------------- | ------------------- | ------------------------------------ |
| File outline              | `outline_utils.py`  | `lean_file_outline`                  |
| Local source search       | `search_utils.py`   | `lean_local_search`                  |
| Loogle (local + remote)   | `loogle.py`         | `lean_loogle`                        |
| Lean REPL                 | `repl.py`           | `lean_multi_attempt`, `lean_run_code`|
| Axiom/source verification | `verify.py`         | `lean_verify`                        |
| `lean --profile` runner   | `profile_utils.py`  | `lean_profile_proof`                 |
| Generic helpers / errors  | `utils.py`          | everywhere                           |
| LSP client / path policy  | `client_utils.py`   | every file-based tool                |

A helper module is plain Python — no MCP imports. It accepts already-validated arguments and either returns a value or raises `LeanToolError` (imported from `lean_lsp_mcp.utils`).

## 3. Register the tool — `server.py`

The decorator + signature is the contract. Template:

```python
@mcp.tool(
    "lean_my_tool",                           # public name (snake_case, prefix lean_)
    annotations=ToolAnnotations(
        title="My Tool",                      # short human label
        readOnlyHint=True,                    # False if it mutates anything
        destructiveHint=False,                # True only for things like lean_build
        idempotentHint=True,                  # repeated calls -> same result
        openWorldHint=False,                  # True if it hits the public internet
    ),
)
# Optional: only for tools that hit a rate-limited remote service.
# @rate_limited("my_tool", max_requests=3, per_seconds=30)
async def my_tool(                            # `async def` for I/O; plain `def` is fine for pure LSP/local calls
    ctx: Context,                             # always first
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line (1-indexed)", ge=1)],
    flag: Annotated[
        bool, Field(description="What this flag toggles")
    ] = False,
) -> MyToolResult:
    """One-line summary shown to the model. Mention key behavior or example.

    Optional second paragraph with examples or warnings (e.g. "SLOW", "use sparingly").
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        _raise_invalid_path(file_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)

    # Convert 1-indexed -> 0-indexed at the boundary
    try:
        return do_the_work(client, rel_path, line - 1, flag=flag)
    except ValueError as exc:
        raise LeanToolError(str(exc)) from exc
```

Some rules:

- **Name starts with `lean_`** and the decorator name is the public identifier. The Python function name can differ from the decorator name (only the decorator name is exposed).
- **`ctx: Context` is always the first parameter.** The `@rate_limited` decorator depends on this.
- **Every non-`ctx` parameter is `Annotated[T, Field(description=..., ...)]`** with sensible bounds (`ge=1` for 1-indexed positions, etc.). Optional params get a default; required params don't.
- **Return a Pydantic model**, not a dict. Multi-shape outputs use a `Union` of two models (see `lean_diagnostic_messages` returning `DiagnosticsResult | InteractiveDiagnosticsResult`).
- **Errors → `raise LeanToolError(...)`**, never bare `Exception` or returning an error string. Wrap underlying exceptions with `from exc`.
- **Path handling** goes through `setup_client_for_file` / `resolve_file_path` / `_raise_invalid_path` from `client_utils`. Don't re-implement project-root inference.
- **Shared state** (LSP client, project path, REPL, build coordinator, loogle manager) is reached via `ctx.request_context.lifespan_context` — never via globals.
- **Progress** for long-running tools: `await _safe_report_progress(ctx, progress=..., total=..., message=...)`.
- **1-indexed in, 0-indexed inside** when calling `leanclient`.
- **Docstring first line is the description shown to the model.** Keep it under ~120 chars; put examples on later lines.

### When the tool needs new lifespan state

If your tool needs something that should live for the whole session (a subprocess, a cache, a new rate-limit bucket), edit `AppContext` and `app_lifespan` in `server.py`:

```python
@dataclass
class AppContext:
    ...
    my_thing: MyThing | None = None        # add the field

# inside app_lifespan, after computing lean_project_path:
my_thing = await initialise_my_thing(lean_project_path)
context = AppContext(..., my_thing=my_thing)

# in the finally block, clean it up
if context and context.my_thing:
    await context.my_thing.close()
```

For a new rate-limit category, also add the empty list to the `rate_limit` dict in `app_lifespan` and apply `@rate_limited("my_category", ...)` below `@mcp.tool`.

## 4. Test it — `tests/`

Tests use the in-process MCP client from `tests/helpers/mcp_client.py`. Add to an existing `test_*.py` if the area fits, or create `tests/test_my_area.py`. Template:

```python
from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.mark.asyncio
async def test_my_tool(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_my_tool",
            {"file_path": str(test_project_path / "Some.lean"), "line": 3},
        )
        assert "expected substring" in result_text(result)
```

Fixtures `mcp_client_factory`, `test_project_path` come from `tests/conftest.py`. Use `result_text(...)` for stringly assertions and access `result.structuredContent` (or the parsed model) for typed assertions — see `tests/test_structured_output.py` for examples.

## 5. Document it — `docs/tools.md`

Add a `#### lean_my_tool` section under the right group ("File interactions (LSP)", "Search", etc.) with one paragraph and, ideally, a `<details>` block showing example output. Match the style of the existing entries.

---

## Optional: runtime configuration

These are handled centrally by `apply_tool_configuration` in `server.py`. They work automatically — just having a registered tool is enough:

- **Disable at runtime**: users set `LEAN_MCP_DISABLED_TOOLS=lean_my_tool,lean_other`.
- **Override description at runtime**: users set `LEAN_MCP_TOOL_DESCRIPTIONS='{"lean_my_tool":"new docstring"}'`.

## Worked example: `lean_file_outline`

A small, end-to-end example showing how the pieces line up. Permalinks pin to commit [`4b5f44d`](https://github.com/oOo0oOo/lean-lsp-mcp/tree/4b5f44d7100ae997f9bec8d2a4707b31ce5d4b13) — line numbers may drift on later commits.

**1. Return model** — [`src/lean_lsp_mcp/models.py` lines 109–129](https://github.com/oOo0oOo/lean-lsp-mcp/blob/4b5f44d7100ae997f9bec8d2a4707b31ce5d4b13/src/lean_lsp_mcp/models.py#L109-L129) defines `OutlineEntry` and the wrapper `FileOutline`:

```python
class OutlineEntry(BaseModel):
    name: str = Field(description="Declaration name")
    kind: str = Field(description="Declaration kind (Thm, Def, Class, Struct, Ns, Ex)")
    start_line: int = Field(description="Start line (1-indexed)")
    end_line: int = Field(description="End line (1-indexed)")
    type_signature: Optional[str] = Field(None, description="Type signature if available")
    children: List["OutlineEntry"] = Field(default_factory=list, description="Nested declarations")


class FileOutline(BaseModel):
    imports: List[str] = Field(default_factory=list, description="Import statements")
    declarations: List[OutlineEntry] = Field(default_factory=list, description="Top-level declarations")
    total_declarations: Optional[int] = Field(None, description="Total count (set when truncated)")
```

Note the recursive `children: List["OutlineEntry"]` (forward reference as a string), the wrapper pattern (`FileOutline` holds a list field rather than being a bare list), and `Optional[...]` with a `None` default rather than `| None`.

**2. Helper module** — heavy lifting lives in [`src/lean_lsp_mcp/outline_utils.py`](https://github.com/oOo0oOo/lean-lsp-mcp/blob/4b5f44d7100ae997f9bec8d2a4707b31ce5d4b13/src/lean_lsp_mcp/outline_utils.py), exposing a single `generate_outline_data(client, rel_path, max_declarations) -> FileOutline`. No MCP imports — just plain Python plus `leanclient`.

**3. Tool registration** — [`src/lean_lsp_mcp/server.py` lines 736–760](https://github.com/oOo0oOo/lean-lsp-mcp/blob/4b5f44d7100ae997f9bec8d2a4707b31ce5d4b13/src/lean_lsp_mcp/server.py#L736-L760):

```python
@mcp.tool(
    "lean_file_outline",
    annotations=ToolAnnotations(
        title="File Outline",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def file_outline(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    max_declarations: Annotated[
        Optional[int], Field(description="Max declarations to return", ge=1)
    ] = None,
) -> FileOutline:
    """Get imports and declarations with type signatures. Token-efficient."""
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        _raise_invalid_path(file_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    return generate_outline_data(client, rel_path, max_declarations)
```

What to notice:

- The function is `def`, not `async def` — it only does in-process LSP calls, no awaiting needed.
- The decorator name (`"lean_file_outline"`) is the public identifier; the Python function is named `file_outline`.
- `ctx: Context` first, `file_path` required, `max_declarations` optional with `Optional[int]` + `None` default + `ge=1` bound.
- Path validation through `setup_client_for_file` / `_raise_invalid_path` — no project-root logic in the tool body.
- The body is three lines: validate path, fetch client from lifespan context, delegate to the helper.

**4. Test** — [`tests/test_outline.py`](https://github.com/oOo0oOo/lean-lsp-mcp/blob/4b5f44d7100ae997f9bec8d2a4707b31ce5d4b13/tests/test_outline.py) calls `client.call_tool("lean_file_outline", {...})` against a real Lean project under `tests/test_project/`.

**5. User-facing docs** — [`docs/tools.md` "File interactions (LSP)" section](https://github.com/oOo0oOo/lean-lsp-mcp/blob/4b5f44d7100ae997f9bec8d2a4707b31ce5d4b13/docs/tools.md#L3-L5).

---

## Quick mental model

A tool is: **decorator metadata** (name + `ToolAnnotations`) + **typed signature** (`ctx` first, then `Annotated[..., Field(...)]` params) + **Pydantic return model from `models.py`** + **one-line docstring** + **body that uses `ctx.request_context.lifespan_context`, validates paths, raises `LeanToolError`, and delegates real work to a helper module**.

Everything else — in particular rate limiting, progress reporting, runtime disabling, description overrides — is opt-in infrastructure that already exists.

import re
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from threading import Lock

from leanclient.aio import AsyncLeanLSPClient, ScratchPool
from leanclient.utils import SYMBOL_KIND_MAP

from lean_lsp_mcp.models import FileOutline, OutlineEntry

METHOD_KIND = {6, "method"}
KIND_TAGS = {"namespace": "Ns"}
_OUTLINE_CACHE_MAX_SIZE = 32
_OUTLINE_CACHE_LOCK = Lock()
_OUTLINE_CACHE: "OrderedDict[tuple[str, str, str, int, int], FileOutline]" = (
    OrderedDict()
)


def _outline_cache_key(
    client: AsyncLeanLSPClient, path: str, content: str
) -> tuple[str, str, str, int, int]:
    project_path = Path(getattr(client, "project_path", ""))
    digest = sha256(content.encode("utf-8")).hexdigest()
    try:
        stat = (project_path / path).stat()
    except OSError:
        return (str(project_path), path, digest, -1, -1)
    return (str(project_path), path, digest, stat.st_mtime_ns, stat.st_size)


def _copy_outline(outline: FileOutline) -> FileOutline:
    return outline.model_copy(deep=True)


def _get_cached_outline(
    key: tuple[str, str, str, int, int],
) -> FileOutline | None:
    with _OUTLINE_CACHE_LOCK:
        outline = _OUTLINE_CACHE.get(key)
        if outline is None:
            return None
        _OUTLINE_CACHE.move_to_end(key)
        return outline


def _store_cached_outline(
    key: tuple[str, str, str, int, int], outline: FileOutline
) -> None:
    with _OUTLINE_CACHE_LOCK:
        _OUTLINE_CACHE[key] = _copy_outline(outline)
        _OUTLINE_CACHE.move_to_end(key)
        while len(_OUTLINE_CACHE) > _OUTLINE_CACHE_MAX_SIZE:
            _OUTLINE_CACHE.popitem(last=False)


def _with_max_declarations(
    outline: FileOutline, max_declarations: int | None
) -> FileOutline:
    result = _copy_outline(outline)
    if max_declarations and len(result.declarations) > max_declarations:
        result.total_declarations = len(result.declarations)
        result.declarations = result.declarations[:max_declarations]
    return result


async def _get_info_trees(
    pool: ScratchPool, content: str, symbols: list[dict]
) -> dict[str, str]:
    """Run a scratch copy with #info_trees insertions; the real doc is untouched."""
    if not symbols:
        return {}

    lines = content.splitlines()
    symbol_by_line = {}
    for i, sym in enumerate(sorted(symbols, key=lambda s: s["range"]["start"]["line"])):
        src_line = sym["range"]["start"]["line"]
        insert_at = src_line + i  # account for previously inserted lines
        lines.insert(insert_at, "#info_trees in")
        symbol_by_line[insert_at] = sym["name"]

    trial = await pool.run_text("\n".join(lines) + "\n")

    info_trees = {}
    for diag in trial.diagnostics.items:
        r = diag.get("fullRange") or diag.get("range") or {}
        diag_line = r.get("start", {}).get("line")
        if diag.get("severity") == 3 and diag_line in symbol_by_line:
            info_trees[symbol_by_line[diag_line]] = diag.get("message", "")

    return info_trees


def _extract_type(info: str, name: str) -> str | None:
    """Extract type signature from info tree message."""
    if m := re.search(
        rf"  • \[Term\] {re.escape(name)} \(isBinder := true\) : ([^@]+) @", info
    ):
        return m.group(1).strip()
    return None


def _extract_fields(info: str, name: str) -> list[tuple[str, str]]:
    """Extract structure/class fields from info tree message."""
    fields = []
    for pattern in [rf"{re.escape(name)}\.(\w+)", rf"@{re.escape(name)}\.(\w+)"]:
        for m in re.finditer(
            rf"  • \[Term\] {pattern} \(isBinder := true\) : (.+?) @", info
        ):
            field_name, full_type = m.groups()
            # Clean up the type signature
            if "]" in full_type:
                field_type = full_type[full_type.rfind("]") + 1 :].lstrip("→ ").strip()
            elif " → " in full_type:
                field_type = full_type.split(" → ")[-1].strip()
            else:
                field_type = full_type.strip()
            fields.append((field_name, field_type))
    return fields


def _strip_line_comment(text: str) -> str:
    """Drop a trailing `--` comment, ignoring one inside a string literal.

    The `--` must be at the start or follow whitespace, which keeps `/--` (a
    doc comment) and any `--` inside an identifier out of it.
    """
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and in_string:
            index += 2
            continue
        if char == '"':
            in_string = not in_string
        elif (
            not in_string
            and char == "-"
            and text.startswith("--", index)
            and (index == 0 or text[index - 1].isspace())
        ):
            return text[:index].rstrip()
        index += 1
    return text


# `let`/`letI`/`haveI` each spend one `:=` inside a statement, before the one
# that starts the declaration body.
# A `let x ← e` in `do` notation binds without `:=`, so it is not counted.
_ASSIGNMENT_OR_BINDER_RE = re.compile(
    r"(?P<binder>(?<![A-Za-z0-9_\'])(?:let|letI|haveI)\s(?!(?:(?!:=|;).)*←))|:="
)

# Lines that end a signature which has no body `:=` of its own. A match arm
# (`| 0 => ...`) begins an equation-compiled body; checking for `=>` keeps an
# absolute value (`|x| ≤ 1`) that opens a continuation line in the signature.
_MATCH_ARM_RE = re.compile(r"^\|.*=>")
# `where` introduces a structure-instance body or auxiliary definitions.
_WHERE_RE = re.compile(r"(?<![A-Za-z0-9_\'.])where(?![A-Za-z0-9_\'])")
# The next declaration or command: the current one is over, whatever it was.
# `#check` and friends count only at column 0: an indented `#` is cardinality
# notation (`#s ≤ n`) continuing the signature.
_HASH_COMMAND_RE = re.compile(r"^#[A-Za-z_]")
_NEXT_COMMAND_RE = re.compile(
    r"^(?:@\[|/--|"
    r"(?:(?:private|protected|noncomputable|partial|unsafe|nonrec)\s+)*"
    r"(?:theorem|lemma|def|abbrev|instance|example|structure|class|inductive"
    r"|axiom|opaque|namespace|section|end|open|variable|universe|attribute"
    r"|set_option|mutual|macro|syntax|notation|elab)\b)"
)


def _scan_assignments(text: str, pending: int = 0) -> tuple[int | None, int]:
    """Find the body `:=` in *text*, carrying unspent binders across calls.

    *pending* counts statement binders seen earlier whose own `:=` has not
    appeared yet. Returns the body index (or None) and the updated count, so a
    caller reading line by line never rescans what it has already seen.
    """
    for match in _ASSIGNMENT_OR_BINDER_RE.finditer(text):
        if match.group("binder"):
            pending += 1
        elif pending:
            pending -= 1
        else:
            return match.start(), pending
    return None, pending


def _body_assignment_index(text: str) -> int | None:
    """Index of the `:=` that begins the declaration body, or None.

    A statement may bind values of its own -- `let m := n + 1` ahead of the
    conclusion -- and each binding spends one `:=`. Splitting on the first one
    cuts the type at that binder and hides everything after it, including the
    conclusion the caller is reading the outline for.
    """
    return _scan_assignments(text)[0]


def _signature_text(lines: list[str], start: int, end: int) -> str | None:
    """Source text of the declaration header at *start*, up to its body.

    The header ends at the body `:=`, at a `where`, or before a match arm.
    Reaching another command first means no header was recognised, and None
    is returned rather than text that belongs to something else.
    """
    parts: list[str] = []
    pending = 0
    for index in range(start, end):
        line = _strip_line_comment(lines[index].strip())
        if index > start:
            if not line or line.startswith(("--", "/-")):
                continue
            if _MATCH_ARM_RE.match(line):
                return " ".join(parts)
            if _NEXT_COMMAND_RE.match(line) or _HASH_COMMAND_RE.match(lines[index]):
                return None
        if (where := _WHERE_RE.search(line)) is not None:
            line = line[: where.start()]
        body_at, pending = _scan_assignments(line, pending)
        if body_at is not None:
            parts.append(line[:body_at])
            return " ".join(parts)
        parts.append(line)
        if where is not None:
            return " ".join(parts)
    return None


def _extract_declarations(content: str, start: int, end: int) -> list[dict]:
    """Extract theorem/lemma/def declarations from file content."""
    lines = content.splitlines()
    decls, i = [], start
    stop = min(end, len(lines))

    while i < stop:
        line = lines[i].strip()
        for keyword in ["theorem", "lemma", "def"]:
            if line.startswith(f"{keyword} "):
                name = line[len(keyword) :].strip().split()[0]
                if name and not name.startswith("_"):
                    # Each line sheds its trailing comment first: a comment
                    # belongs to its physical line, and joining before stripping
                    # would let it swallow the rest of the signature.
                    header = _signature_text(lines, i, stop)
                    type_sig = None
                    if header is not None:
                        # Everything before the body, minus keyword and name.
                        sig_part = header.strip()[len(keyword) :].strip()
                        if sig_part.startswith(name):
                            type_sig = sig_part[len(name) :].strip()

                    decls.append(
                        {
                            "name": name,
                            "kind": "method",
                            "range": {
                                "start": {"line": i, "character": 0},
                                "end": {"line": i, "character": len(lines[i])},
                            },
                            "_keyword": keyword,
                            "_type": type_sig,
                        }
                    )
                break
        i += 1
    return decls


def _flatten_symbols(
    symbols: list[dict], indent: int = 0, content: str = ""
) -> list[tuple[dict, int]]:
    """Recursively flatten symbol hierarchy, extracting declarations from namespaces."""
    result = []
    for sym in symbols:
        result.append((sym, indent))
        children = sym.get("children", [])

        # Extract theorem/lemma/def from namespace bodies
        if content and sym.get("kind") == "namespace":
            ns_range = sym["range"]
            ns_start = ns_range["start"]["line"]
            ns_end = ns_range["end"]["line"]
            children = children + _extract_declarations(content, ns_start, ns_end)

        if children:
            result.extend(_flatten_symbols(children, indent + 1, content))
    return result


def _detect_tag(
    name: str, kind: str, type_sig: str, has_fields: bool, keyword: str | None
) -> str:
    """Determine the appropriate tag for a symbol."""
    if has_fields:
        return "Class" if "→" in type_sig else "Struct"
    if name == "example":
        return "Ex"
    if keyword in {"theorem", "lemma"}:
        return "Thm"
    if type_sig and any(marker in type_sig for marker in ["∀", "="]):
        return "Thm"
    if type_sig and "→" in type_sig.replace(" → ", "", 1):  # More than one arrow
        return "Thm"
    return KIND_TAGS.get(kind, "Def")


def _build_outline_entry(
    sym: dict, type_sigs: dict, fields_map: dict, indent: int
) -> OutlineEntry | None:
    """Build a structured outline entry for a symbol."""
    name = sym["name"]
    type_sig = sym.get("_type") or type_sigs.get(name, "")
    fields = fields_map.get(name, [])

    tag = _detect_tag(
        name, sym.get("kind", ""), type_sig, bool(fields), sym.get("_keyword")
    )
    start = sym["range"]["start"]["line"] + 1
    end = sym["range"]["end"]["line"] + 1

    # Add fields as children for structs/classes
    children = [
        OutlineEntry(
            name=fname,
            kind="field",
            start_line=start,
            end_line=start,
            type_signature=ftype,
            children=[],
        )
        for fname, ftype in fields
    ]

    return OutlineEntry(
        name=name,
        kind=tag,
        start_line=start,
        end_line=end,
        type_signature=type_sig if type_sig else None,
        children=children,
    )


async def generate_outline_data(
    client: AsyncLeanLSPClient,
    pool: ScratchPool,
    path: str,
    max_declarations: int | None = None,
) -> FileOutline:
    """Generate structured outline data for a Lean file."""
    await client.reload_from_disk(path)
    content = client.content(path)
    cache_key = _outline_cache_key(client, path, content)
    if cached := _get_cached_outline(cache_key):
        return _with_max_declarations(cached, max_declarations)

    # Extract imports (handles both 'import X' and 'public import X')
    imports = []
    for line in content.splitlines():
        s = line.strip()
        if s.startswith("public import "):
            imports.append(s[14:])
        elif s.startswith("import "):
            imports.append(s[7:])

    symbols = await client.document_symbols(path)
    # Match legacy leanclient behavior: top-level kinds as strings.
    for symbol in symbols:
        if isinstance(symbol.get("kind"), int):
            symbol["kind"] = SYMBOL_KIND_MAP.get(symbol["kind"], "unknown")
    if not symbols and not imports:
        outline = FileOutline(imports=[], declarations=[])
        _store_cached_outline(cache_key, outline)
        return _with_max_declarations(outline, max_declarations)

    # Flatten symbol tree and extract namespace declarations
    all_symbols = _flatten_symbols(symbols, content=content)

    # Get info trees only for LSP symbols (not extracted declarations)
    lsp_methods = [
        s
        for s, _ in all_symbols
        if s.get("kind") in METHOD_KIND and "_keyword" not in s
    ]
    info_trees = await _get_info_trees(pool, content, lsp_methods)

    # Extract type signatures and fields from info trees
    type_sigs = {
        name: sig
        for name, info in info_trees.items()
        if (sig := _extract_type(info, name))
    }
    fields_map = {
        name: fields
        for name, info in info_trees.items()
        if (fields := _extract_fields(info, name))
    }

    # Build declarations list
    declarations = []
    for sym, indent in all_symbols:
        if (
            sym.get("kind") in METHOD_KIND
            or sym.get("_keyword")
            or sym.get("kind") == "namespace"
        ):
            entry = _build_outline_entry(sym, type_sigs, fields_map, indent)
            if entry:
                declarations.append(entry)

    outline = FileOutline(imports=imports, declarations=declarations)
    _store_cached_outline(cache_key, outline)
    return _with_max_declarations(outline, max_declarations)

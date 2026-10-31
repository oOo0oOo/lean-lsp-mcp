import re
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from threading import Lock

from leanclient.aio import AsyncLeanLSPClient
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


def _strip_line_comment(text: str) -> str:
    """Drop a trailing `--` comment, ignoring one inside a string literal.

    The `--` must be at the start or follow whitespace, which keeps `/--` (a
    doc comment) and any `--` inside an identifier out of it.
    """
    if "--" not in text:
        return text
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
_DECLARATION_START_RE = re.compile(
    r"^(?:(?:@\[[^\]]*\]\s*)|"
    r"(?:(?:private|protected|noncomputable|partial|unsafe|nonrec)\s+))*"
    r"(?P<keyword>theorem|lemma|def)\s+(?P<name>[^\s:]+)"
)
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
        declaration = _DECLARATION_START_RE.match(line)
        if declaration:
            keyword = declaration.group("keyword")
            name = declaration.group("name")
            if name and not name.startswith("_"):
                # Each line sheds its trailing comment first: a comment
                # belongs to its physical line, and joining before stripping
                # would let it swallow the rest of the signature.
                header = _signature_text(lines, i, stop)
                type_sig = None
                if header is not None:
                    # Everything before the body, minus attributes, keyword,
                    # and name.
                    header_declaration = _DECLARATION_START_RE.match(header.strip())
                    if header_declaration:
                        type_sig = header.strip()[header_declaration.end() :].strip()

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


def _fields_from_symbols(symbols: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """Read structure/class fields and details already present in LSP symbols."""
    fields_map: dict[str, list[tuple[str, str]]] = {}

    def visit(symbol: dict) -> None:
        fields = []
        for child in symbol.get("children", []):
            if child.get("kind") not in {7, 8, "property", "field"}:
                continue
            name = child.get("name", "").rsplit(".", 1)[-1]
            detail = child.get("detail", "")
            if detail.startswith(f"{name} : "):
                detail = detail[len(name) + 3 :]
            fields.append((name, detail))
        if fields:
            fields_map[symbol.get("name", "")] = fields
        for child in symbol.get("children", []):
            visit(child)

    for symbol in symbols:
        visit(symbol)
    return fields_map


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
    type_sig = sym.get("_type") or type_sigs.get(name, "") or sym.get("detail", "")
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

    # Document-symbol requests are barrier-gated by default. Lean can answer
    # from the current document snapshot, so don't wait for full elaboration.
    symbols = await client.document_symbols(path, fresh=False)
    # Match legacy leanclient behavior: top-level kinds as strings.
    for symbol in symbols:
        if isinstance(symbol.get("kind"), int):
            symbol["kind"] = SYMBOL_KIND_MAP.get(symbol["kind"], "unknown")
    # Flatten symbol tree and extract declarations from namespaces.
    all_symbols = _flatten_symbols(symbols, content=content)

    # Document symbols omit `lemma` declarations in some Lean/Mathlib files.
    # The source scanner also supplies signatures for declarations when Lean
    # has not finished elaborating them yet. Merge by source range so a symbol
    # already reported by the LSP is enriched instead of duplicated.
    for text_decl in _extract_declarations(content, 0, len(content.splitlines())):
        decl_line = text_decl["range"]["start"]["line"]
        decl_name = text_decl["name"].rsplit(".", 1)[-1]
        match = next(
            (
                sym
                for sym, _ in all_symbols
                if sym.get("kind") != "namespace"
                and sym.get("name", "").rsplit(".", 1)[-1] == decl_name
                and sym.get("range", {}).get("start", {}).get("line", -1)
                <= decl_line
                <= sym.get("range", {}).get("end", {}).get("line", -1)
            ),
            None,
        )
        if match is None:
            all_symbols.append((text_decl, 0))
        else:
            match["_keyword"] = text_decl["_keyword"]
            if text_decl["_type"]:
                match["_type"] = text_decl["_type"]

    type_sigs: dict[str, str] = {}
    fields_map = _fields_from_symbols(symbols)

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
    # Do not freeze a syntax-only snapshot in the cache while Lean is still
    # producing symbol details. A later call can then pick up signatures that
    # have become available, without making the first call wait for them.
    has_unreported_scoped_symbols = not symbols and bool(
        re.search(
            r"^\s*(?:namespace|class|structure|inductive|instance)\b", content, re.M
        )
    )
    has_unresolved_details = any(
        (sym.get("kind") in METHOD_KIND or sym.get("_keyword"))
        and not (sym.get("_type") or sym.get("detail"))
        for sym, _ in all_symbols
    )
    if not has_unreported_scoped_symbols and not has_unresolved_details:
        _store_cached_outline(cache_key, outline)
    return _with_max_declarations(outline, max_declarations)

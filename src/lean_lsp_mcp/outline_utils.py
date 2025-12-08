import re
from typing import Dict, List, Optional, Tuple
from leanclient import LeanLSPClient
from leanclient.utils import DocumentContentChange

from lean_lsp_mcp.models import FileOutline, OutlineEntry


METHOD_KIND = {6, "method"}
KIND_TAGS = {"namespace": "Ns"}


def _get_info_trees(
    client: LeanLSPClient, path: str, symbols: List[Dict]
) -> Dict[str, str]:
    """Insert #info_trees commands, collect diagnostics, then revert changes."""
    if not symbols:
        return {}

    symbol_by_line = {}
    changes = []
    for i, sym in enumerate(sorted(symbols, key=lambda s: s["range"]["start"]["line"])):
        line = sym["range"]["start"]["line"] + i
        symbol_by_line[line] = sym["name"]
        changes.append(DocumentContentChange("#info_trees in\n", [line, 0], [line, 0]))

    client.update_file(path, changes)
    diagnostics = client.get_diagnostics(path)

    info_trees = {
        symbol_by_line[diag["range"]["start"]["line"]]: diag["message"]
        for diag in diagnostics
        if diag["severity"] == 3 and diag["range"]["start"]["line"] in symbol_by_line
    }

    # Revert in reverse order
    client.update_file(
        path,
        [
            DocumentContentChange("", [line, 0], [line + 1, 0])
            for line in sorted(symbol_by_line.keys(), reverse=True)
        ],
    )
    return info_trees


def _extract_type(info: str, name: str) -> Optional[str]:
    """Extract type signature from info tree message."""
    if m := re.search(
        rf"  • \[Term\] {re.escape(name)} \(isBinder := true\) : ([^@]+) @", info
    ):
        return m.group(1).strip()
    return None


def _extract_fields(info: str, name: str) -> List[Tuple[str, str]]:
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


def _extract_declarations(content: str, start: int, end: int) -> List[Dict]:
    """Extract theorem/lemma/def declarations from file content."""
    lines = content.splitlines()
    decls, i = [], start

    while i < min(end, len(lines)):
        line = lines[i].strip()
        for keyword in ["theorem", "lemma", "def"]:
            if line.startswith(f"{keyword} "):
                name = line[len(keyword) :].strip().split()[0]
                if name and not name.startswith("_"):
                    # Collect until :=
                    decl_lines = [line]
                    j = i + 1
                    while j < min(end, len(lines)) and ":=" not in " ".join(decl_lines):
                        if (next_line := lines[j].strip()) and not next_line.startswith(
                            "--"
                        ):
                            decl_lines.append(next_line)
                        j += 1

                    # Extract signature (everything before :=, minus keyword and name)
                    full_decl = " ".join(decl_lines)
                    type_sig = None
                    if ":=" in full_decl:
                        sig_part = (
                            full_decl.split(":=", 1)[0].strip()[len(keyword) :].strip()
                        )
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
    symbols: List[Dict], indent: int = 0, content: str = ""
) -> List[Tuple[Dict, int]]:
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
    name: str, kind: str, type_sig: str, has_fields: bool, keyword: Optional[str]
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


def _format_symbol(sym: Dict, type_sigs: Dict, fields_map: Dict, indent: int) -> str:
    """Format a single symbol with its type signature and fields."""
    name = sym["name"]
    type_sig = sym.get("_type") or type_sigs.get(name, "")
    fields = fields_map.get(name, [])

    tag = _detect_tag(
        name, sym.get("kind", ""), type_sig, bool(fields), sym.get("_keyword")
    )
    prefix = "\t" * indent

    start = sym["range"]["start"]["line"] + 1
    end = sym["range"]["end"]["line"] + 1
    line_info = f"L{start}" if start == end else f"L{start}-{end}"

    result = f"{prefix}[{tag}: {line_info}] {name}"
    if type_sig:
        result += f" : {type_sig}"

    for fname, ftype in fields:
        result += f"\n{prefix}\t{fname} : {ftype}"

    return result + "\n"


def _build_outline_entry(
    sym: Dict, type_sigs: Dict, fields_map: Dict, indent: int
) -> Optional[OutlineEntry]:
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


def generate_outline_data(client: LeanLSPClient, path: str) -> FileOutline:
    """Generate structured outline data for a Lean file."""
    client.open_file(path)
    content = client.get_file_content(path)

    # Extract imports
    imports = [
        line.strip()[7:]
        for line in content.splitlines()
        if line.strip().startswith("import ")
    ]

    symbols = client.get_document_symbols(path)
    if not symbols and not imports:
        return FileOutline(imports=[], declarations=[])

    # Flatten symbol tree and extract namespace declarations
    all_symbols = _flatten_symbols(symbols, content=content)

    # Get info trees only for LSP symbols (not extracted declarations)
    lsp_methods = [
        s
        for s, _ in all_symbols
        if s.get("kind") in METHOD_KIND and "_keyword" not in s
    ]
    info_trees = _get_info_trees(client, path, lsp_methods)

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

    return FileOutline(imports=imports, declarations=declarations)


def generate_outline(client: LeanLSPClient, path: str) -> str:
    """Generate a concise outline of a Lean file showing structure and signatures."""
    client.open_file(path)
    content = client.get_file_content(path)

    # Extract imports
    imports = [
        line.strip()[7:]
        for line in content.splitlines()
        if line.strip().startswith("import ")
    ]

    symbols = client.get_document_symbols(path)
    if not symbols and not imports:
        return f"# {path}\n\n*No symbols or imports found*\n"

    # Flatten symbol tree and extract namespace declarations
    all_symbols = _flatten_symbols(symbols, content=content)

    # Get info trees only for LSP symbols (not extracted declarations)
    lsp_methods = [
        s
        for s, _ in all_symbols
        if s.get("kind") in METHOD_KIND and "_keyword" not in s
    ]
    info_trees = _get_info_trees(client, path, lsp_methods)

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

    # Build output
    parts = []
    if imports:
        parts.append("## Imports\n" + "\n".join(imports))

    if symbols:
        declarations = [
            _format_symbol(sym, type_sigs, fields_map, indent)
            for sym, indent in all_symbols
            if sym.get("kind") in METHOD_KIND
            or sym.get("_keyword")
            or sym.get("kind") == "namespace"
        ]
        parts.append("## Declarations\n" + "".join(declarations).rstrip())

    return "\n\n".join(parts) + "\n"

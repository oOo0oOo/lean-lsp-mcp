from dataclasses import dataclass


@dataclass(frozen=True)
class SplitSnippet:
    header: str
    body: str
    header_line_count: int


def split_snippet(code: str) -> SplitSnippet:
    """
    Splits a code snippet into a header (imports) and body.

    - Header: all lines at the top that are 'import ...' or blank before the first non-import line.
      If any import starts with 'import Mathlib', include a single 'import Mathlib' at the top of the header.
      Other imports follow in their original order, without duplicates.
    - Body: the rest of the code starting from the first non-import/non-blank line.
    """
    lines = code.splitlines()

    # Separate header from body
    i = 0
    while i < len(lines) and (
        lines[i].strip() == "" or lines[i].strip().startswith("import ")
    ):
        i += 1
    header_lines = [x.strip() for x in lines[:i]]
    body = "\n".join(lines[i:])

    # Process imports in header
    import_lines = [line for line in header_lines if line.startswith("import ")]
    imports: list[str] = []
    seen: set[str] = set()
    has_mathlib = False
    for line in import_lines:
        if line.startswith("import Mathlib"):
            has_mathlib = True
        else:
            if line not in seen:
                seen.add(line)
                imports.append(line)

    # Build final header
    result_header: list[str] = []
    if has_mathlib:
        result_header.append("import Mathlib")
    result_header.extend(imports)

    header = "\n".join(result_header)
    return SplitSnippet(header=header, body=body, header_line_count=i)

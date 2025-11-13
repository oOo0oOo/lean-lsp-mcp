import os
import secrets
import sys
import tempfile
from typing import List, Dict, Optional

from mcp.server.auth.provider import AccessToken, TokenVerifier


class OutputCapture:
    """Capture any output to stdout and stderr at the file descriptor level."""

    def __init__(self):
        self.original_stdout_fd = None
        self.original_stderr_fd = None
        self.temp_file = None
        self.captured_output = ""

    def __enter__(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8"
        )
        self.original_stdout_fd = os.dup(sys.stdout.fileno())
        self.original_stderr_fd = os.dup(sys.stderr.fileno())
        os.dup2(self.temp_file.fileno(), sys.stdout.fileno())
        os.dup2(self.temp_file.fileno(), sys.stderr.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.original_stdout_fd, sys.stdout.fileno())
        os.dup2(self.original_stderr_fd, sys.stderr.fileno())
        os.close(self.original_stdout_fd)
        os.close(self.original_stderr_fd)

        self.temp_file.flush()
        self.temp_file.seek(0)
        self.captured_output = self.temp_file.read()
        self.temp_file.close()
        os.unlink(self.temp_file.name)

    def get_output(self):
        return self.captured_output


class OptionalTokenVerifier(TokenVerifier):
    """Minimal verifier that accepts a single pre-shared token."""

    def __init__(self, expected_token: str):
        self._expected_token = expected_token

    async def verify_token(self, token: str | None) -> AccessToken | None:
        if token is None or not secrets.compare_digest(token, self._expected_token):
            return None
        # AccessToken requires both client_id and scopes parameters to be provided.
        return AccessToken(token=token, client_id="lean-lsp-mcp-optional", scopes=[])


def format_diagnostics(diagnostics: List[Dict], select_line: int = -1) -> List[str]:
    """Format the diagnostics messages.

    Args:
        diagnostics (List[Dict]): List of diagnostics.
        select_line (int): If -1, format all diagnostics. If >= 0, only format diagnostics for this line.

    Returns:
        List[str]: Formatted diagnostics messages.
    """
    msgs = []
    if select_line != -1:
        diagnostics = filter_diagnostics_by_position(diagnostics, select_line, None)

    # Format more compact
    for diag in diagnostics:
        r = diag.get("fullRange", diag.get("range", None))
        if r is None:
            r_text = "No range"
        else:
            r_text = f"l{r['start']['line'] + 1}c{r['start']['character'] + 1}-l{r['end']['line'] + 1}c{r['end']['character'] + 1}"
        msgs.append(f"{r_text}, severity: {diag['severity']}\n{diag['message']}")
    return msgs


def format_goal(goal, default_msg):
    if goal is None:
        return default_msg
    rendered = goal.get("rendered")
    return rendered.replace("```lean\n", "").replace("\n```", "") if rendered else None


def _utf16_index_to_py_index(text: str, utf16_index: int) -> int | None:
    """Convert an LSP UTF-16 column index into a Python string index."""
    if utf16_index < 0:
        return None

    units = 0
    for idx, ch in enumerate(text):
        code_point = ord(ch)
        next_units = units + (2 if code_point > 0xFFFF else 1)

        if utf16_index < next_units:
            return idx
        if utf16_index == next_units:
            return idx + 1

        units = next_units
    if units >= utf16_index:
        return len(text)
    return None


def extract_range(content: str, range: dict) -> str:
    """Extract the text from the content based on the range.

    Args:
        content (str): The content to extract from.
        range (dict): The range to extract.

    Returns:
        str: The extracted range text.
    """
    start_line = range["start"]["line"]
    start_char = range["start"]["character"]
    end_line = range["end"]["line"]
    end_char = range["end"]["character"]

    lines = content.splitlines(keepends=True)
    if not lines:
        lines = [""]

    line_offsets: List[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    total_length = len(content)

    def position_to_offset(line: int, character: int) -> int | None:
        if line == len(lines) and character == 0:
            return total_length
        if line < 0 or line >= len(lines):
            return None
        py_index = _utf16_index_to_py_index(lines[line], character)
        if py_index is None:
            return None
        if py_index > len(lines[line]):
            return None
        return line_offsets[line] + py_index

    start_offset = position_to_offset(start_line, start_char)
    end_offset = position_to_offset(end_line, end_char)

    if start_offset is None or end_offset is None or start_offset > end_offset:
        return "Range out of bounds"

    return content[start_offset:end_offset]


def find_start_position(content: str, query: str) -> dict | None:
    """Find the position of the query in the content.

    Args:
        content (str): The content to search in.
        query (str): The query to find.

    Returns:
        dict | None: The position of the query in the content. {"line": int, "column": int}
    """
    lines = content.splitlines()
    for line_number, line in enumerate(lines):
        char_index = line.find(query)
        if char_index != -1:
            return {"line": line_number, "column": char_index}
    return None


def format_line(
    file_content: str,
    line_number: int,
    column: Optional[int] = None,
    cursor_tag: Optional[str] = "<cursor>",
) -> str:
    """Show a line and cursor position in a file.

    Args:
        file_content (str): The content of the file.
        line_number (int): The line number (1-indexed).
        column (Optional[int]): The column number (1-indexed). If None, no cursor position is shown.
        cursor_tag (Optional[str]): The tag to use for the cursor position. Defaults to "<cursor>".
    Returns:
        str: The formatted position.
    """
    lines = file_content.splitlines()
    line_number -= 1
    if line_number < 0 or line_number >= len(lines):
        return "Line number out of range"
    line = lines[line_number]
    if column is None:
        return line
    column -= 1
    # Allow placing the cursor at end-of-line (column == len(line))
    if column < 0 or column > len(line):
        return "Invalid column number"
    return f"{line[:column]}{cursor_tag}{line[column:]}"


def filter_diagnostics_by_position(
    diagnostics: List[Dict], line: Optional[int], column: Optional[int]
) -> List[Dict]:
    """Return diagnostics that intersect the requested (0-indexed) position."""

    if line is None:
        return list(diagnostics)

    matches: List[Dict] = []
    for diagnostic in diagnostics:
        diagnostic_range = diagnostic.get("range") or diagnostic.get("fullRange")
        if not diagnostic_range:
            continue

        start = diagnostic_range.get("start", {})
        end = diagnostic_range.get("end", {})
        start_line = start.get("line")
        end_line = end.get("line")

        if start_line is None or end_line is None:
            continue
        if line < start_line or line > end_line:
            continue

        start_char = start.get("character")
        end_char = end.get("character")

        if column is None:
            if (
                line == end_line
                and line != start_line
                and end_char is not None
                and end_char == 0
            ):
                continue
            matches.append(diagnostic)
            continue

        if start_char is None:
            start_char = 0
        if end_char is None:
            end_char = column + 1

        if start_line == end_line and start_char == end_char:
            if column == start_char:
                matches.append(diagnostic)
            continue

        if line == start_line and column < start_char:
            continue
        if line == end_line and column >= end_char:
            continue

        matches.append(diagnostic)

    return matches


def search_symbols(symbols: List[Dict], target_name: str) -> Dict | None:
    """Recursively search through symbols and their children.

    Args:
        symbols: List of LSP document symbols
        target_name: Name of the symbol to find (case-sensitive)

    Returns:
        The matching symbol dict, or None if not found
    """
    for symbol in symbols:
        if symbol.get("name") == target_name:
            return symbol
        # Search nested declarations (children)
        children = symbol.get("children", [])
        if children:
            result = search_symbols(children, target_name)
            if result:
                return result
    return None


def get_declaration_range(
    client, file_path: str, declaration_name: str
) -> tuple[int, int] | None:
    """Get the line range (1-indexed) of a declaration by name using LSP document symbols.

    Args:
        client: The Lean LSP client instance (LeanLSPClient)
        file_path: Relative path to the Lean file
        declaration_name: Name of the declaration to find (case-sensitive)

    Returns:
        Tuple of (start_line, end_line) as 1-indexed integers, or None if not found
    """
    from lean_lsp_mcp.server import logger

    try:
        # Ensure file is opened (LSP needs this to analyze the file)
        client.open_file(file_path)

        # Get document symbols from LSP
        symbols = client.get_document_symbols(file_path)

        if not symbols:
            logger.debug(
                "No document symbols returned for '%s' - file may not be processed yet",
                file_path,
            )
            return None

        matching_symbol = search_symbols(symbols, declaration_name)
        if not matching_symbol:
            return None

        # Extract range - LSP returns 0-indexed, convert to 1-indexed
        range_info = matching_symbol.get("range")
        if not range_info:
            return None

        start_line = range_info["start"]["line"] + 1
        end_line = range_info["end"]["line"] + 1

        return (start_line, end_line)

    except Exception as e:
        logger.warning(
            "Failed to get declaration range for '%s' in '%s': %s",
            declaration_name,
            file_path,
            e,
        )
        return None

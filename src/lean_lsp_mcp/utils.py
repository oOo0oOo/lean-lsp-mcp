import os
import sys
import tempfile
from typing import List, Dict, Optional

from mcp.server.auth.provider import AccessToken, TokenVerifier


class StdoutToStderr:
    """Redirects stdout to stderr at the file descriptor level bc lake build logging"""

    def __init__(self):
        self.original_stdout_fd = None

    def __enter__(self):
        self.original_stdout_fd = os.dup(sys.stdout.fileno())
        stderr_fd = sys.stderr.fileno()
        os.dup2(stderr_fd, sys.stdout.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_stdout_fd is not None:
            os.dup2(self.original_stdout_fd, sys.stdout.fileno())
            os.close(self.original_stdout_fd)
            self.original_stdout_fd = None


class OutputCapture:
    """Capture any output to stdout and stderr at the file descriptor level."""

    def __init__(self):
        self.original_stdout_fd = None
        self.original_stderr_fd = None
        self.temp_file = None
        self.captured_output = ""

    def __enter__(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
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

    lines = content.splitlines()
    if start_line < 0 or end_line >= len(lines):
        return "Range out of bounds"
    if start_line == end_line:
        return lines[start_line][start_char:end_char]
    else:
        selected_lines = lines[start_line : end_line + 1]
        selected_lines[0] = selected_lines[0][start_char:]
        selected_lines[-1] = selected_lines[-1][:end_char]
        return "\n".join(selected_lines)


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
    if line_number < 1 or line_number > len(lines):
        return "Line number out of range"
    line = lines[line_number]
    if column is None:
        return line
    column -= 1
    if column < 0 or column >= len(lines):
        return "Invalid column number"
    return f"{line[:column]}{cursor_tag}{line[column:]}"


def filter_diagnostics_by_position(
    diagnostics: List[Dict], line: int, column: Optional[int]
) -> List[Dict]:
    """Find diagnostics at a specific position.

    Args:
        diagnostics (List[Dict]): List of diagnostics.
        line (int): The line number (0-indexed).
        column (Optional[int]): The column number (0-indexed).

    Returns:
        List[Dict]: List of diagnostics at the specified position.
    """
    if column is None:
        return [
            d
            for d in diagnostics
            if d["range"]["start"]["line"] <= line <= d["range"]["end"]["line"]
        ]

    return [
        d
        for d in diagnostics
        if d["range"]["start"]["line"] <= line <= d["range"]["end"]["line"]
        and d["range"]["start"]["character"] <= column < d["range"]["end"]["character"]
    ]


class OptionalTokenVerifier(TokenVerifier):
    def __init__(self, expected_token: str):
        self.expected_token = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if token == self.expected_token:
            return AccessToken(token=token, client_id="lean-lsp-mcp", scopes=["user"])
        return None

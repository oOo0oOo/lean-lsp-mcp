import os
import sys
from typing import List, Dict


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


def format_diagnostics(diagnostics: List[Dict], select_line: int = -1) -> List[str]:
    """Format the diagnostics messages.

    Args:
        diagnostics (List[Dict]): List of diagnostics.
        select_line (int): If -1, format all diagnostics. If >= 0, only format diagnostics for this line.

    Returns:
        List[str]: Formatted diagnostics messages.
    """
    msgs = []
    # Format more compact
    for diag in diagnostics:
        r = diag.get("fullRange", diag.get("range", None))
        if r is None:
            r_text = "No range"
        elif select_line >= 0 and r["start"]["line"] != select_line:
            continue
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

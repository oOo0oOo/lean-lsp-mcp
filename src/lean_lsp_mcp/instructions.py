INSTRUCTIONS = """## General Rules
- All line and column numbers are 1-indexed (use lean_file_contents if unsure).
- Always analyze/search context before each file edit.
- This MCP does NOT make permanent file changes. Use other tools for editing.
- Work iteratively: Small steps, intermediate sorries, frequent checks.

## Key Tools
- lean_goal: Check proof state. USE OFTEN!
- lean_diagnostic_messages: Understand the current proof situation.
- lean_hover_info: Documentation about terms and lean syntax.
- lean_leansearch: Search theorems using natural language or Lean terms.
- lean_loogle: Search definitions and theorems by name, type, or subexpression.
- lean_state_search: Search theorems using goal-based search.
"""

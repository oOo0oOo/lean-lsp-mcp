INSTRUCTIONS = """## General Rules
- All line and column numbers are 1-indexed.
- This MCP does NOT edit files. Use other tools for editing.
- Work iteratively: small steps, intermediate sorries, frequent lean_goal checks.

## Tool Quick Reference

### Core Tools (no rate limit)
- **lean_goal**: Proof state at position. Omit `column` to see before/after tactic effect. USE OFTEN!
  - "no goals" = proof complete at that point. You're done!
- **lean_diagnostic_messages**: Compiler errors/warnings. Check after edits.
  - "no goals to be solved" = remove extraneous tactics
- **lean_hover_info**: Type signature + docs. Column must be at START of identifier.
- **lean_completions**: IDE autocomplete. Use on incomplete code (after `.` or partial name).
- **lean_local_search**: Fast local declaration search. Use BEFORE trying a lemma name.
- **lean_file_outline**: Token-efficient file skeleton. Slow-ish but very useful.
- **lean_multi_attempt**: Test multiple tactics without editing. Pass list like ["simp", "ring", "omega"].
- **lean_declaration_file**: Get source where symbol is declared. Use sparingly (large output).
- **lean_run_code**: Run standalone snippet. Use rarely - prefer editing actual files.
- **lean_build**: Rebuild project + restart LSP. Only if needed (new imports). SLOW!

### Search Tools (rate limited: 3 req/30s, except leanfinder: 10 req/30s)
- **lean_leansearch**: Natural language → mathlib. "sum of even numbers is even"
- **lean_loogle**: Type pattern → mathlib. `(?a → ?b) → List ?a → List ?b` finds map
- **lean_leanfinder**: Semantic/conceptual search. "commutativity of addition"
- **lean_state_search**: Goal → closing lemmas. Call at sorry position.
- **lean_hammer_premise**: Goal → premises for simp/omega/aesop.

## Search Tool Decision Tree
1. "Does X exist in my project?" → lean_local_search (instant, local)
2. "I need a lemma that says X" → lean_leansearch (natural language)
3. "Find lemma matching type pattern" → lean_loogle (e.g., `Real.sin`, `|- _ < _`)
4. "What's the Lean name for this concept?" → lean_leanfinder (semantic)
5. "What lemma closes this goal?" → lean_state_search (at sorry)
6. "What to feed simp/omega?" → lean_hammer_premise

After finding a name, verify with lean_local_search, then lean_hover_info for full signature.

## Query Examples

### lean_loogle patterns
- Constant: `Real.sin` → lemmas mentioning Real.sin
- Name substring: `"comm"` → lemmas with "comm" in name
- Type shape: `(?a → ?b) → List ?a → List ?b` → finds List.map
- Subexpression: `_ * (_ ^ _)` → products with powers
- Conclusion: `|- _ < _ → _ + 1 < _ + 1` → inequalities

### lean_leansearch queries
- "injective function has left inverse"
- "Cauchy-Schwarz inequality"
- Lean-ish: "{f : A → B} (hf : Injective f) : ∃ g, LeftInverse g f"

## Proof Workflow
1. lean_goal at sorry (omit column for before/after view)
2. Search for lemmas (decision tree above)
3. lean_multi_attempt to test candidates
4. Edit file externally
5. lean_diagnostic_messages to verify
6. lean_goal again - if goals_after is "no goals", proof is COMPLETE. Stop!
7. Repeat until done

## Common Errors
- "unknown identifier X" → lean_local_search "X", check imports
- "type mismatch" → compare expected/actual types in message
- "failed to synthesize instance" → add instance with haveI/letI
- "no goals to be solved" → remove extra tactics

## Return Formats
All list-returning tools return JSON arrays. Single-result tools return JSON objects.
Empty results return `[]`. Parse with standard JSON.
"""

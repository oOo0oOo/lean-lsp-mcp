### File interactions (LSP)

#### lean_file_outline

Get a concise outline of a Lean file showing imports and declarations with type signatures (theorems, definitions, classes, structures).

#### lean_diagnostic_messages

Get all diagnostic messages for a Lean file. This includes infos, warnings and errors. Use `severity` (`"error"`, `"warning"`, `"info"`, `"hint"`) to filter to a single level. `interactive=True` returns verbose nested `TaggedText` with embedded widgets. For "Try This" suggestions, prefer `lean_code_actions`.

<details>
<summary>Example output</summary>

```
l20c42-l20c46, severity: 1
simp made no progress

l21c11-l21c45, severity: 1
function expected at
  h_empty
term has type
  T ∩ compl T = ∅

...
```
</details>

#### lean_goal

Get the proof goal at a specific location (line or line & column) in a Lean file.

<details>
<summary>Example output (line)</summary>

```
Before:
S : Type u_1
inst✝¹ : Fintype S
inst✝ : Nonempty S
P : Finset (Set S)
hPP : ∀ T ∈ P, ∀ U ∈ P, T ∩ U ≠ ∅
hPS : ¬∃ T ∉ P, ∀ U ∈ P, T ∩ U ≠ ∅
compl : Set S → Set S := fun T ↦ univ \ T
hcompl : ∀ T ∈ P, compl T ∉ P
all_subsets : Finset (Set S) := Finset.univ
h_comp_in_P : ∀ T ∉ P, compl T ∈ P
h_partition : ∀ (T : Set S), T ∈ P ∨ compl T ∈ P
⊢ P.card = 2 ^ (Fintype.card S - 1)
After:
no goals
```
</details>

#### lean_term_goal

Get the term goal at a specific position (line & column) in a Lean file.

#### lean_hover_info

Retrieve hover information (documentation) for symbols, terms, and expressions in a Lean file (at a specific line & column).

<details>
<summary>Example output (hover info on a `sorry`)</summary>

```
The `sorry` tactic is a temporary placeholder for an incomplete tactic proof,
closing the main goal using `exact sorry`.

This is intended for stubbing-out incomplete parts of a proof while still having a syntactically correct proof skeleton.
Lean will give a warning whenever a proof uses `sorry`, so you aren't likely to miss it,
but you can double check if a theorem depends on `sorry` by looking for `sorryAx` in the output
of the `#print axioms my_thm` command, the axiom used by the implementation of `sorry`.
```
</details>

#### lean_declaration_file

Get the file contents where a symbol or term is declared.

#### lean_references

Find all references to a symbol at a given position (line & column), including its declaration.

#### lean_completions

Code auto-completion: Find available identifiers or import suggestions at a specific position (line & column) in a Lean file.

#### lean_run_code

Run/compile an independent Lean code snippet/file and return the result or error message.
<details>
<summary>Example output (code snippet: `#eval 5 * 7 + 3`)</summary>

```
l1c1-l1c6, severity: 3
38
```
</details>

#### lean_multi_attempt

Attempt multiple tactics at a proof position and return goal state and diagnostics for each.
Useful to screen different proof attempts before committing to one.

Accepts:
- `line` to target a tactic line
- optional `column` to target an exact source position, similar to `lean_goal`

Execution mode:
- When `LEAN_REPL=true` and `column` is omitted, uses the REPL tactic mode for up to 5x faster execution (see [Environment Variables](#environment-variables)).
- When `column` is provided, uses the LSP path so the attempt is evaluated at the exact source position.

<details>
<summary>Example output (attempting `rw [Nat.pow_sub (Fintype.card_pos_of_nonempty S)]` and `by_contra h_neq`)</summary>

```
  rw [Nat.pow_sub (Fintype.card_pos_of_nonempty S)]:
S : Type u_1
inst✝¹ : Fintype S
inst✝ : Nonempty S
P : Finset (Set S)
hPP : ∀ T ∈ P, ∀ U ∈ P, T ∩ U ≠ ∅
hPS : ¬∃ T ∉ P, ∀ U ∈ P, T ∩ U ≠ ∅
⊢ P.card = 2 ^ (Fintype.card S - 1)

l14c7-l14c51, severity: 1
unknown constant 'Nat.pow_sub'

  by_contra h_neq:
 S : Type u_1
inst✝¹ : Fintype S
inst✝ : Nonempty S
P : Finset (Set S)
hPP : ∀ T ∈ P, ∀ U ∈ P, T ∩ U ≠ ∅
hPS : ¬∃ T ∉ P, ∀ U ∈ P, T ∩ U ≠ ∅
h_neq : ¬P.card = 2 ^ (Fintype.card S - 1)
⊢ False

...
```
</details>

#### lean_code_actions

Get LSP code actions for a line. Returns resolved edits for "Try This" suggestions (`simp?`, `exact?`, `apply?`) and other quick fixes. The agent applies the edits using its own editing tools.

<details>
<summary>Example output (line with <code>simp?</code>)</summary>

```json
{
  "actions": [
    {
      "title": "Try this: simp only [zero_add]",
      "is_preferred": false,
      "edits": [
        {
          "new_text": "simp only [zero_add]",
          "start_line": 3,
          "start_column": 37,
          "end_line": 3,
          "end_column": 42
        }
      ]
    }
  ]
}
```
</details>

#### lean_get_widgets

Get panel widgets at a position (proof visualizations, `#html`, custom widgets). Returns raw widget data - may be verbose.

<details>
<summary>Example output (<code>#html</code> widget)</summary>

```json
{
  "widgets": [
    {
      "id": "ProofWidgets.HtmlDisplayPanel",
      "javascriptHash": "15661785739548337049",
      "props": {
        "html": {
          "element": ["b", [], [{"text": "Hello widget"}]]
        }
      },
      "range": {
        "start": {"line": 4, "character": 0},
        "end": {"line": 4, "character": 50}
      }
    }
  ]
}
```
</details>

#### lean_get_widget_source

Get the JavaScript source code of a widget by its `javascriptHash` (from `lean_get_widgets` or `lean_diagnostic_messages` with `interactive=True`). Useful for understanding custom widget rendering logic. Returns full JS module - may be verbose.

#### lean_profile_proof

Profile a theorem to identify slow tactics. Runs `lean --profile` on an isolated copy of the theorem and returns per-line timing data.

<details>
<summary>Example output (profiling a theorem using simp)</summary>

```json
{
  "ms": 42.5,
  "lines": [
    {"line": 7, "ms": 38.2, "text": "simp [add_comm, add_assoc]"}
  ],
  "categories": {
    "simp": 35.1,
    "typeclass inference": 4.2
  }
}
```
</details>

#### lean_verify

Check theorem soundness: returns axioms used + optional source pattern scan for `unsafe`, `set_option debug.*`, `@[implemented_by]`, etc. Standard axioms are `propext`, `Classical.choice`, `Quot.sound` - anything else (e.g. `sorryAx`) indicates an unsound proof. Source warnings require [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`).

<details>
<summary>Example output (theorem using sorry)</summary>

```json
{
  "axioms": ["propext", "sorryAx"],
  "warnings": [
    {"line": 5, "pattern": "set_option debug.skipKernelTC"}
  ]
}
```
</details>

#### lean_minimal_hypotheses

For each explicit `(h : T)` hypothesis of a theorem, drop it and re-elaborate the file via the LSP. Reports which hypotheses are load-bearing and which are actually unused. For load-bearing hypotheses, the verdict includes a list of `breaks` — every new error caused by removing the binder, with line, column, and message — so you can see *where* in the proof the dropped hypothesis was used. Useful for "minimum hypotheses needed" / counterfactual reasoning when sharpening a result.

Skips implicit `{x : α}` and instance `[inst : C]` binders. Does not rewrite the proof body — a body that names `h` will fail to elaborate without the binder, which is the truthful answer (load-bearing).

Slow: each hypothesis triggers a full re-elaboration capped at `inactivity_timeout` (default 60 s). The original file content is restored before the tool returns.

<details>
<summary>Example output (one of two hypotheses unused)</summary>

```json
{
  "theorem_name": "minhyp_one_unused",
  "file": "MinimalHypothesesTest.lean",
  "verdicts": [
    {
      "binder": "(h1 : 1 + 1 = 2)",
      "status": "load-bearing",
      "breaks": [
        {"severity": "error", "message": "unknown identifier 'h1'", "line": 7, "column": 3}
      ],
      "detail": ""
    },
    {
      "binder": "(h2 : 2 + 2 = 4)",
      "status": "removable",
      "breaks": [],
      "detail": ""
    }
  ],
  "skipped_implicit": 0
}
```
</details>

### Local Search Tools

#### lean_local_search

Search for Lean definitions and theorems in the local Lean project and stdlib.
This is useful to confirm declarations actually exist and prevent hallucinating APIs.

This tool requires [ripgrep](https://github.com/BurntSushi/ripgrep?tab=readme-ov-file#installation) (`rg`) to be installed and available in your PATH.

### External Search Tools

Currently most external tools are separately **rate limited to 3 requests per 30 seconds**. Please don't ruin the fun for everyone by overusing these amazing free services!

Please cite the original authors of these tools if you use them!

#### lean_leansearch

Search for theorems in Mathlib using [leansearch.net](https://leansearch.net) (natural language search).

[Github Repository](https://github.com/frenzymath/LeanSearch) | [Arxiv Paper](https://arxiv.org/abs/2403.13310)

- Supports natural language, mixed queries, concepts, identifiers, and Lean terms.
- Example: `bijective map from injective`, `n + 1 <= m if n < m`, `Cauchy Schwarz`, `List.sum`, `{f : A → B} (hf : Injective f) : ∃ h, Bijective h`

<details>
<summary>Example output (query by LLM: `bijective map from injective`)</summary>

```json
  {
    "module_name": "Mathlib.Logic.Function.Basic",
    "kind": "theorem",
    "name": "Function.Bijective.injective",
    "signature": " {f : α → β} (hf : Bijective f) : Injective f",
    "type": "∀ {α : Sort u_1} {β : Sort u_2} {f : α → β}, Function.Bijective f → Function.Injective f",
    "value": ":= hf.1",
    "informal_name": "Bijectivity Implies Injectivity",
    "informal_description": "For any function $f \\colon \\alpha \\to \\beta$, if $f$ is bijective, then $f$ is injective."
  },
  ...
```
</details>

#### lean_loogle

Search for Lean definitions and theorems using [loogle.lean-lang.org](https://loogle.lean-lang.org/).

[Github Repository](https://github.com/nomeata/loogle)

- Supports queries by constant, lemma name, subexpression, type, or conclusion.
- Example: `Real.sin`, `"differ"`, `_ * (_ ^ _)`, `(?a -> ?b) -> List ?a -> List ?b`, `|- tsum _ = _ * tsum _`
- **Local mode available**: Use `--loogle-local` to run loogle locally (avoids rate limits, see [Local Loogle](#local-loogle) section)
- **Self-hosted**: Set `LOOGLE_URL` to point at a self-hosted instance. Set `LOOGLE_HEADERS` to a JSON object for extra headers (e.g. `'{"X-API-Key": "..."}'`).

<details>
<summary>Example output (`Real.sin`)</summary>

```json
[
  {
    "type": " (x : ℝ) : ℝ",
    "name": "Real.sin",
    "module": "Mathlib.Data.Complex.Trigonometric"
  },
  ...
]
```
</details>

#### lean_leanfinder

Semantic search for Mathlib theorems using [Lean Finder](https://huggingface.co/spaces/delta-lab-ai/Lean-Finder).

[Arxiv Paper](https://arxiv.org/abs/2510.15940)

- Supports informal descriptions, user questions, proof states, and statement fragments.
- Examples: `algebraic elements x,y over K with same minimal polynomial`, `Does y being a root of minpoly(x) imply minpoly(x)=minpoly(y)?`, `⊢ |re z| ≤ ‖z‖` + `transform to squared norm inequality`, `theorem restrict Ioi: restrict Ioi e = restrict Ici e`

<details>
<summary>Example output</summary>

Query: `Does y being a root of minpoly(x) imply minpoly(x)=minpoly(y)?`

```json
  [
    [
      "/-- If `y : L` is a root of `minpoly K x`, then `minpoly K y = minpoly K x`. -/\ntheorem eq_of_root {x y : L} (hx : IsAlgebraic K x)\n    (h_ev : Polynomial.aeval y (minpoly K x) = 0) : minpoly K y = minpoly K x :=\n  ((eq_iff_aeval_minpoly_eq_zero hx.isIntegral).mpr h_ev).symm",
      
      "Let $L/K$ be a field extension, and let $x, y \\in L$ be elements such that $y$ is a root of the minimal polynomial of $x$ over $K$. If $x$ is algebraic over $K$, then the minimal polynomial of $y$ over $K$ is equal to the minimal polynomial of $x$ over $K$, i.e., $\\text{minpoly}_K(y) = \\text{minpoly}_K(x)$. This means that if $y$ satisfies the polynomial equation defined by $x$, then $y$ shares the same minimal polynomial as $x$."
    ],
    ...
  ]
```
</details>

#### lean_state_search

Search for applicable theorems for the current proof goal using [premise-search.com](https://premise-search.com/).

[Github Repository](https://github.com/ruc-ai4math/Premise-Retrieval) | [Arxiv Paper](https://arxiv.org/abs/2501.13959)

A self-hosted version is [available](https://github.com/ruc-ai4math/LeanStateSearch) and encouraged. You can set an environment variable `LEAN_STATE_SEARCH_URL` to point to your self-hosted instance. It defaults to `https://premise-search.com`.

Uses the first goal at a given line and column.
Returns a list of relevant theorems.
<details> <summary>Example output (line 24, column 3)</summary>

```json
[
  {
    "name": "Nat.mul_zero",
    "formal_type": "∀ (n : Nat), n * 0 = 0",
    "module": "Init.Data.Nat.Basic"
  },
  ...
]
```
</details>


#### lean_hammer_premise

Search for relevant premises based on the current proof state using the [Lean Hammer Premise Search](https://github.com/hanwenzhu/lean-premise-server).

[Github Repository](https://github.com/hanwenzhu/lean-premise-server) | [Arxiv Paper](https://arxiv.org/abs/2506.07477)

A self-hosted version is [available](https://github.com/hanwenzhu/lean-premise-server) and encouraged. You can set an environment variable `LEAN_HAMMER_URL` to point to your self-hosted instance. It defaults to `http://leanpremise.net`.

Uses the first goal at a given line and column.
Returns a list of relevant premises (theorems) that can be used to prove the goal.

Note: We use a simplified version, [LeanHammer](https://github.com/JOSHCLUNE/LeanHammer) might have better premise search results.
<details><summary>Example output (line 24, column 3)</summary>

```json
[
  "MulOpposite.unop_injective",
  "MulOpposite.op_injective",
  "WellFoundedLT.induction",
  ...
]
```
</details>

### Project-level tools

#### lean_build

Rebuild the Lean project and restart the Lean LSP server.
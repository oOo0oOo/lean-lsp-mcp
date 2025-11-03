<h1 align="center">
  lean-lsp-mcp
</h1>

<h3 align="center">Lean Theorem Prover MCP</h3>

<p align="center">
  <a href="https://pypi.org/project/lean-lsp-mcp/">
    <img src="https://img.shields.io/pypi/v/lean-lsp-mcp.svg" alt="PyPI version" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/oOo0oOo/lean-lsp-mcp" alt="last update" />
  </a>
  <a href="https://github.com/oOo0oOo/lean-lsp-mcp/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/oOo0oOo/lean-lsp-mcp.svg" alt="license" />
  </a>
</p>

MCP server that allows agentic interaction with the [Lean theorem prover](https://lean-lang.org/) via the [Language Server Protocol](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/) using [leanclient](https://github.com/oOo0oOo/leanclient). This server provides a range of tools for LLM agents to understand, analyze and interact with Lean projects.

## Key Features

* **Rich Lean Interaction**: Access diagnostics, goal states, term information, hover documentation and more.
* **External Search Tools**: Use `LeanSearch`, `Loogle`, `Lean Finder`, `Lean Hammer` and `Lean State Search` to find relevant theorems and definitions.
* **Easy Setup**: Simple configuration for various clients, including VSCode, Cursor and Claude Code.

## Setup

### Overview

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), a Python package manager.
2. Make sure your Lean project builds quickly by running `lake build` manually.
3. Configure your IDE/Setup
4. (Optional, highly recommended) Install [ripgrep](https://github.com/BurntSushi/ripgrep?tab=readme-ov-file#installation) (`rg`) to reduce hallucinations using local search.

### 1. Install uv

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) for your system. On Linux/MacOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 2. Run `lake build`

`lean-lsp-mcp` will run `lake serve` in the project root to use the language server (for most tools). Some clients (e.g. Cursor) might timeout during this process. Therefore, it is recommended to run `lake build` manually before starting the MCP. This ensures a faster build time and avoids timeouts.

### 3. Configure your IDE/Setup

<details>
<summary><b>VSCode</b></summary>
One-click config setup:

[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_Server-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=lean-lsp&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22lean-lsp-mcp%22%5D%7D)

[![Install in VS Code Insiders](https://img.shields.io/badge/VS_Code_Insiders-Install_Server-24bfa5?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=lean-lsp&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22lean-lsp-mcp%22%5D%7D&quality=insiders)

OR using the setup wizard:

Ctrl+Shift+P > "MCP: Add Server..." > "Command (stdio)" > "uvx lean-lsp-mcp" > "lean-lsp" (or any name you like) > Global or Workspace

OR manually adding config by opening `mcp.json` with: 

Ctrl+Shift+P > "MCP: Open User Configuration"

and adding the following

```jsonc
{
    "servers": {
        "lean-lsp": {
            "type": "stdio",
            "command": "uvx",
            "args": [
                "lean-lsp-mcp"
            ]
        }
    }
}
```

If you installed VSCode on Windows and are using WSL2 as your development environment, you may need to use this config instead:

```jsonc
{
    "servers": {
        "lean-lsp": {
            "type": "stdio",
            "command": "wsl.exe",
            "args": [
                "uvx",
                "lean-lsp-mcp"
            ]
        }
    }
}
```
If that doesn't work, you can try cloning this repository and replace `"lean-lsp-mcp"` with `"/path/to/cloned/lean-lsp-mcp"`.

</details>

<details>
<summary><b>Cursor</b></summary>
1. Open MCP Settings (File > Preferences > Cursor Settings > MCP)

2. "+ Add a new global MCP Server" > ("Create File")

3. Paste the server config into `mcp.json` file:

```jsonc
{
    "mcpServers": {
        "lean-lsp": {
            "command": "uvx",
            "args": ["lean-lsp-mcp"]
        }
    }
}
```
</details>

<details>
<summary><b>Claude Code</b></summary>
Run one of these commands in the root directory of your Lean project (where `lakefile.toml` is located):

```bash
# Local-scoped MCP server
claude mcp add lean-lsp uvx lean-lsp-mcp

# OR project-scoped MCP server
# (creates or updates a .mcp.json file in the current directory)
claude mcp add lean-lsp -s project uvx lean-lsp-mcp
```

You can find more details about MCP server configuration for Claude Code [here](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#configure-mcp-servers).
</details>

#### Claude Skill: Lean4 Theorem Proving

If you are using [Claude Desktop](https://modelcontextprotocol.io/quickstart/user) or [Claude Code](https://claude.ai/code), you can also install the [Lean4 Theorem Proving Skill](https://github.com/cameronfreer/lean4-skills/tree/main/plugins/lean4-theorem-proving). This skill provides additional prompts and templates for interacting with Lean4 projects and includes a section on interacting with the `lean-lsp-mcp` server.

### 4. Install ripgrep (optional but recommended)

For the local search tool `lean_local_search`, install [ripgrep](https://github.com/BurntSushi/ripgrep?tab=readme-ov-file#installation) (`rg`) and make sure it is available in your PATH.

## MCP Tools


### File interactions (LSP)

#### lean_file_contents

Get the contents of a Lean file, optionally with line number annotations.

#### lean_diagnostic_messages

Get all diagnostic messages for a Lean file. This includes infos, warnings and errors.

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

Attempt multiple lean code snippets on a line and return goal state and diagnostics for each snippet.
This tool is useful to screen different proof attempts before using the most promising one.

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

### Disabling Tools

Many clients allow the user to disable specific tools manually (e.g. lean_build).

**VSCode**: Click on the Wrench/Screwdriver icon in the chat.

**Cursor**: In "Cursor Settings" > "MCP" click on the name of a tool to disable it (strikethrough).

## MCP Configuration

This MCP server works out-of-the-box without any configuration. However, a few optional settings are available.

### Environment Variables

- `LEAN_LOG_LEVEL`: Log level for the server. Options are "INFO", "WARNING", "ERROR", "NONE". Defaults to "INFO".
- `LEAN_PROJECT_PATH`: Path to your Lean project root. Set this if the server cannot automatically detect your project.
- `LEAN_LSP_MCP_TOKEN`: Secret token for bearer authentication when using `streamable-http` or `sse` transport.
- `LEAN_STATE_SEARCH_URL`: URL for a self-hosted [premise-search.com](https://premise-search.com) instance.
- `LEAN_HAMMER_URL`: URL for a self-hosted [Lean Hammer Premise Search](https://github.com/hanwenzhu/lean-premise-server) instance.

You can also often set these environment variables in your MCP client configuration:
<details>
<summary><b>VSCode mcp.json Example</b></summary>

```jsonc
{
    "servers": {
        "lean-lsp": {
            "type": "stdio",
            "command": "uvx",
            "args": [
                "lean-lsp-mcp"
            ],
            "env": {
                "LEAN_PROJECT_PATH": "/path/to/your/lean/project",
                "LEAN_LOG_LEVEL": "NONE"
            }
        }
    }
}
```
</details>

### Transport Methods

The Lean LSP MCP server supports the following transport methods:

- `stdio`: Standard input/output (default)
- `streamable-http`: HTTP streaming
- `sse`: Server-sent events (MCP legacy, use `streamable-http` if possible)

You can specify the transport method using the `--transport` argument when running the server. For `sse` and `streamable-http` you can also optionally specify the host and port:

```bash
uvx lean-lsp-mcp --transport stdio # Default transport
uvx lean-lsp-mcp --transport streamable-http # Available at http://127.0.0.1:8000/mcp
uvx lean-lsp-mcp --transport sse --host localhost --port 12345 # Available at http://localhost:12345/sse
```

### Bearer Token Authentication

Transport via `streamable-http` and `sse` supports bearer token authentication. This allows publicly accessible MCP servers to restrict access to authorized clients.

Set the `LEAN_LSP_MCP_TOKEN` environment variable (or see section 3 for setting env variables in MCP config) to a secret token before starting the server.

Example Linux/MacOS setup:

```bash
export LEAN_LSP_MCP_TOKEN="your_secret_token"
uvx lean-lsp-mcp --transport streamable-http
```

Clients should then include the token in the `Authorization` header.

## Notes on MCP Security

There are many valid security concerns with the Model Context Protocol (MCP) in general!

This MCP server is meant as a research tool and is currently in beta.
While it does not handle any sensitive data such as passwords or API keys, it still includes various security risks:
- Access to your local file system.
- No input or output validation.

Please be aware of these risks. Feel free to audit the code and report security issues!

For more information, you can use [Awesome MCP Security](https://github.com/Puliczek/awesome-mcp-security) as a starting point.

## Development

### MCP Inspector

```bash
npx @modelcontextprotocol/inspector uvx --with-editable path/to/lean-lsp-mcp python -m lean_lsp_mcp.server
```

### Run Tests

```bash
uv sync --all-extras
uv run pytest tests
```

## Publications using lean-lsp-mcp

- Ax-Prover: A Deep Reasoning Agentic Framework for Theorem Proving in Mathematics and Quantum Physics [arxiv](https://arxiv.org/abs/2510.12787)

## Related Projects

- [LeanTool](https://github.com/GasStationManager/LeanTool)
- [LeanExplore MCP](https://www.leanexplore.com/docs/mcp)

## License & Citation

**MIT** licensed. See [LICENSE](LICENSE) for more information.

Citing this repository is highly appreciated but not required by the license.

```bibtex
@software{lean-lsp-mcp,
  author = {Oliver Dressler},
  title = {{Lean LSP MCP: Tools for agentic interaction with the Lean theorem prover}},
  url = {https://github.com/oOo0oOo/lean-lsp-mcp},
  month = {3},
  year = {2025}
}
```

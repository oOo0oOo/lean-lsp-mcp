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

MCP that allows agentic interaction with the [Lean theorem prover](https://lean-lang.org/) via the [Language Server Protocol](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/) using [leanclient](https://github.com/oOo0oOo/leanclient). This server provides a range of tools for AI models to understand, analyze and interact with Lean projects.

**Currently beta testing**: Please help us by submitting bug reports, feedback and feature requests.

## Key Features

* **Rich Lean Interaction**: Access diagnostics, goal states, term information, and hover documentation.
* **Agent-Focused Toolset:** Includes tools for theorem search (leansearch.net), code completion, and project builds.
* **Easy Setup**: Simple configuration for various IDEs, including VSCode and Cursor.

## Setup

### Overview

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), a Python package manager.
2. Make sure your Lean project builds quickly by running `lake build` manually.
3. Add JSON configuration to your IDE/Setup and configure LEAN_PROJECT_PATH.

### 1. Install uv

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) for your system.

E.g. on Linux/MacOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Run `lake build`

`lean-lsp-mcp` will run `lake build` in the project root upon startup. Some IDEs (like Cursor) might timeout during this process. Therefore, it is recommended to run `lake build` manually before starting the MCP. This ensures a faster startup time and avoids timeouts.

E.g. on Linux/MacOS:
```bash
cd /path/to/lean/project
lake build
```

Note: Your build does not necessarily need to be successful, some errors or warnings (e.g. `declaration uses 'sorry'`) are OK.

### 3. a) VSCode Setup

VSCode and VSCode Insiders are supporting MCPs in [agent mode](https://code.visualstudio.com/blogs/2025/04/07/agentMode). For VSCode you might have to enable `Chat > Agent: Enable` in the settings.

1. One-click config setup:

[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_Server-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=lean-lsp&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22lean-lsp-mcp%22%5D%2C%22env%22%3A%7B%22LEAN_PROJECT_PATH%22%3A%22path%2520to%2520lean%2520project%2520root%22%7D%7D)

[![Install in VS Code Insiders](https://img.shields.io/badge/VS_Code_Insiders-Install_Server-24bfa5?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=lean-lsp&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22lean-lsp-mcp%22%5D%2C%22env%22%3A%7B%22LEAN_PROJECT_PATH%22%3A%22path%2520to%2520lean%2520project%2520root%22%7D%7D&quality=insiders)

OR manually add config to `settings.json` (global):

```json
{
    "mcp": {
        "servers": {
            "lean-lsp": {
                "command": "uvx",
                "args": ["lean-lsp-mcp"],
                "env": {
                    "LEAN_PROJECT_PATH": "/path/to/lean/project"
                }
            }
        }
    }
}
```

2. Next change the env variable `LEAN_PROJECT_PATH` to point to the root of your Lean project. This is required for the MCP to work. You can also remove this from the config and set this env variable differently.

3. Click "Start" above server config, open a Lean file, change to agent mode in the chat and run e.g. "auto proof" to get started:

### 3. b) Cursor Setup

1. Open MCP Settings (File > Preferences > Cursor Settings > MCP)

2. "+ Add a new global MCP Server" > ("Create File")

3. Paste the server config into `mcp.json` file and adjust the `LEAN_PROJECT_PATH` to point to the root of your Lean project:

```json
{
    "mcpServers": {
        "lean-lsp": {
            "command": "uvx",
            "args": ["lean-lsp-mcp"],
            "env": {
                "LEAN_PROJECT_PATH": "/path/to/lean/project"
            }
        }
    }
}
```

### 3. c) Claude Code

Run one of these commands in the root directory of your Lean project (where `lakefile.toml` is located):

```bash
# Local-scoped MCP server
claude mcp add lean-lsp uvx lean-lsp-mcp -e LEAN_PROJECT_PATH=$PWD

# OR project-scoped MCP server (creates or updates a .mcp.json file in the current directory)
claude mcp add lean-lsp -s project uvx lean-lsp-mcp -e LEAN_PROJECT_PATH=$PWD
```

You can find more details about MCP server configuration for Claude Code [here](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#configure-mcp-servers).

### Other Setups

Other setups, such as [Claude Desktop](https://modelcontextprotocol.io/quickstart/user) or [OpenAI Agent SDK](https://openai.github.io/openai-agents-python/mcp/) should work with similar configs.

## Tools

Lean LSP MCP currently provides various tools to interact with the Lean theorem prover:

### Meta tools

#### lean_auto_proof_instructions

Get detailed instructions on how to use the Lean LSP MCP to automatically prove theorems. This is a tool call because many clients do not support prompts yet, it is also available as a prompt. You can check out the current instruction prompt in [prompts.py](https://github.com/oOo0oOo/lean-lsp-mcp/blob/main/src/lean_lsp_mcp/prompts.py).

### Core interactions

#### lean_diagnostic_messages

Get all diagnostic messages for a Lean file. This includes infos, warnings and errors.

<details>
<summary>Example output</summary>

l20c42-l20c46, severity: 1<br>
simp made no progress

l21c11-l21c45, severity: 1<br>
function expected at
  h_empty
term has type
  T ∩ compl T = ∅

...
</details>

#### lean_goal

Get the proof goal at a specific location (line or line & column) in a Lean file.

<details>
<summary>Example output (line)</summary>
Before:<br>
S : Type u_1<br>
inst✝¹ : Fintype S<br>
inst✝ : Nonempty S<br>
P : Finset (Set S)<br>
hPP : ∀ T ∈ P, ∀ U ∈ P, T ∩ U ≠ ∅<br>
hPS : ¬∃ T ∉ P, ∀ U ∈ P, T ∩ U ≠ ∅<br>
compl : Set S → Set S := fun T ↦ univ \ T<br>
hcompl : ∀ T ∈ P, compl T ∉ P<br>
all_subsets : Finset (Set S) := Finset.univ<br>
h_comp_in_P : ∀ T ∉ P, compl T ∈ P<br>
h_partition : ∀ (T : Set S), T ∈ P ∨ compl T ∈ P<br>
⊢ P.card = 2 ^ (Fintype.card S - 1)<br>
After:<br>
no goals
</details>

#### lean_term_goal

Get the term goal at a specific position (line & column) in a Lean file.

#### lean_hover_info

Retrieve hover information (documentation) for symbols, terms, and expressions in a Lean file (at a specific line & column).

<details>
<summary>Example output (hover info on a `sorry`)</summary>
The `sorry` tactic is a temporary placeholder for an incomplete tactic proof,<br>
closing the main goal using `exact sorry`.<br><br>

This is intended for stubbing-out incomplete parts of a proof while still having a syntactically correct proof skeleton.<br>
Lean will give a warning whenever a proof uses `sorry`, so you aren't likely to miss it,<br>
but you can double check if a theorem depends on `sorry` by looking for `sorryAx` in the output<br>
of the `#print axioms my_thm` command, the axiom used by the implementation of `sorry`.<br>
</details>

#### lean_declaration_file

Get the file contents where a symbol or term is declared.

#### lean_completions

Code auto-completion: Find available identifiers or import suggestions at a specific position (line & column) in a Lean file.

#### lean_multi_attempt

Attempt multiple lean code snippets on a line and return goal state and diagnostics for each snippet.
This tool is useful to screen different proof attempts before using the most promising one.

<details>
<summary>Example output (attempting `rw [Nat.pow_sub (Fintype.card_pos_of_nonempty S)]` and `by_contra h_neq`)</summary>
  rw [Nat.pow_sub (Fintype.card_pos_of_nonempty S)]:<br>
S : Type u_1<br>
inst✝¹ : Fintype S<br>
inst✝ : Nonempty S<br>
P : Finset (Set S)<br>
hPP : ∀ T ∈ P, ∀ U ∈ P, T ∩ U ≠ ∅<br>
hPS : ¬∃ T ∉ P, ∀ U ∈ P, T ∩ U ≠ ∅<br>
⊢ P.card = 2 ^ (Fintype.card S - 1)<br>
<br>
l14c7-l14c51, severity: 1<br>
unknown constant 'Nat.pow_sub'<br>
<br>
  by_contra h_neq:<br>
 S : Type u_1<br>
inst✝¹ : Fintype S<br>
inst✝ : Nonempty S<br>
P : Finset (Set S)<br>
hPP : ∀ T ∈ P, ∀ U ∈ P, T ∩ U ≠ ∅<br>
hPS : ¬∃ T ∉ P, ∀ U ∈ P, T ∩ U ≠ ∅<br>
h_neq : ¬P.card = 2 ^ (Fintype.card S - 1)<br>
⊢ False<br>
<br>
...
</details>

#### lean_leansearch

Search for theorems in Mathlib using leansearch.net (natural language search).

<details>
<summary>Example output (query by LLM: "finite set, subset, complement, cardinality, half, partition")</summary>
<br>
{"module_name": ["Mathlib", "Data", "Fintype", "Card"], "kind": "theorem", "name": ["Finset", "card_compl"], "signature": " [DecidableEq \u03b1] [Fintype \u03b1] (s : Finset \u03b1) : #s\u1d9c = Fintype.card \u03b1 - #s", "type": "\u2200 {\u03b1 : Type u_1} [inst : DecidableEq \u03b1] [inst_1 : Fintype \u03b1] (s : Finset \u03b1), s\u1d9c.card = Fintype.card \u03b1 - s.card", "value": ":=\n  Finset.card_univ_diff s", "docstring": null, "informal_name": "Cardinality of Complement Set in Finite Type", "informal_description": "For a finite type $\\alpha$ with decidable equality and a finite subset $s \\subseteq \\alpha$, the cardinality of the complement of $s$ equals the difference between the cardinality of $\\alpha$ and the cardinality of $s$, i.e.,\n$$|s^c| = \\text{card}(\\alpha) - |s|.$$"}

...<br>
More answers like above<br>
...
</details>

#### lean_proofs_complete

Check if all proofs in a file are complete. This is currently very simple and will be improved in the future.

### File operations

#### lean_file_contents

Get the contents of a Lean file, optionally with line number annotations.

### Project-level tools

#### lean_build

Rebuild the Lean project and restart the Lean LSP server.

## Prompts

#### lean_auto_proof_instructions

Get detailed instructions on how to use the Lean LSP MCP to automatically prove theorems. See above (Meta tools).

## Example Uses

Here are a few example prompts and interactions to try. All examples use VSCode (Agent Mode) and Gemini 2.5 Pro (Preview).

### Using auto proof prompt

Open unfinished proof. Run prompt "auto proof" in VSCode (Agent Mode) or Cursor will use the `lean_auto_proof_instructions` tool to get detailed instructions on how to use the Lean LSP MCP to automatically prove theorems.

![VS Code Agent Mode](media/auto_proof.png)

### Analyze a theorem

Open `Algebra/Lie/Abelian.lean`. Example prompt:

"Analyze commutative_ring_iff_abelian_lie_ring thoroughly using various tools such as goal, term goal, hover info. Explain the key proof steps in english.".

![Analyzing a theorem in chat](media/analyze_theorem.png)

### Design proof approaches

Open an incomplete proof such as [putnam 1964 b2](https://github.com/trishullab/PutnamBench/blob/main/lean4/src/putnam_1964_b2.lean). Example prompt:

"First analyze the problem statement by checking the goal, hover info and looking up key declarations. Next use up to three queries to leansearch to design three different approaches to solve this problem. Very concisely present each approach and its key challenge."

![Designing proof approaches](media/proof_approaches.png)

## Disabling Tools

Many IDEs allow to disable specific tools manually (e.g. lean_build).

**VSCode**: Click on the Wrench/Screwdriver icon in the chat.

**Cursor**: In "Cursor Settings" > "MCP" click on the name of a tool to disable it (strikethrough).

## Notes on MCP Security

There are many valid security concerns with the Model Context Protocol (MCP) in general!

This MCP is meant as a research tool and is currently in beta.
While it does not handle any sensitive data such as passwords or API keys, it still includes various security risks:
- Access to your local file system.
- No rate limiting on tool calls.
- No input or output validation.

Please be aware of these risks. Feel free to audit the code and report security issues!

For more information, you can use [Awesome MCP Security](https://github.com/Puliczek/awesome-mcp-security) as a starting point.

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
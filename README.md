<h1 align="center">
  lean-lsp-mcp
</h1>

<h4 align="center">Lean Theorem Prover MCP</h4>

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

MCP that allows agentic interaction with the [Lean theorem prover](https://lean-lang.org/) via the [Language Server Protocol](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/) using [leanclient](https://github.com/oOo0oOo/leanclient).

**Currently beta testing**: Please help us by submitting bug reports, feedback and feature requests.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), a Python package manager.
2. Make sure your Lean project builds quickly by running `lake build` manually in a terminal in the project root.
3. Add JSON configuration to your IDE/Setup.
4. Configure env variable LEAN_PROJECT_PATH.

### 1. Install uv

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) for your system.

E.g. on Linux/MacOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Run lake build

`lean-lsp-mcp` will run `lake build` in the project root upon startup. Some IDEs (like Cursor) might timeout during this process. Therefore, it is recommended to run `lake build` manually before starting the MCP. This ensures a faster startup time and avoids timeouts.

E.g. on Linux/MacOS:
```bash
cd /path/to/lean/project
lake build
```

Note: Your build does not necessarily need to be successful, some errors or warnings (e.g. `declaration uses 'sorry'`) are OK.

### 3. VSCode Setup

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

![VS Code Agent Mode](media/vscode_agent_mode.png)


### 3. Cursor Setup

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

4. Open a Lean file and run e.g. "auto proof" in a new chat.


### Other Setups

Other setups, such as [Claude Desktop](https://modelcontextprotocol.io/quickstart/user), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#configure-mcp-servers) or [OpenAI Agent SDK](https://openai.github.io/openai-agents-python/mcp/) should work with similar configs (untested).


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

#### lean_completions

Code auto-completion: Find available identifiers or import suggestions at a specific position (line & column) in a Lean file.

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

#### lean_project_path

Get the path to the current Lean project root directory.

#### lean_lsp_restart

Restart the LSP server and optionally rebuild the Lean project.

## Prompts

#### lean_auto_proof_instructions

Get detailed instructions on how to use the Lean LSP MCP to automatically prove theorems. See above (Meta tools).

## Related Projects

- [LeanTool](https://github.com/GasStationManager/LeanTool)

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
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
  <a href="https://github.com/oOo0oOo/leanclient/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/oOo0oOo/lean-lsp-mcp.svg" alt="license" />
  </a>
</p>

MCP that allows agentic interaction with the [Lean theorem prover](https://lean-lang.org/) via the [Language Server Protocol](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/) using [leanclient](https://github.com/oOo0oOo/leanclient).

**Currently beta testing**: Please help us by submitting bug reports, feedback and feature requests.

## Setup

1. Add JSON configuration to your IDE/Setup.
2. Configure env variable LEAN_PROJECT_PATH.

### VSCode Insiders

VSCode Insiders (bleeding edge version of VSCode) has introduced [agent mode](https://code.visualstudio.com/blogs/2025/02/24/introducing-copilot-agent-mode) in February 2025. This feature will eventually be released in the stable version of VSCode.

1. One-click config setup:

[<img alt="Install in VS Code Insiders" src="https://img.shields.io/badge/VS_Code_Insiders-VS_Code_Insiders?style=flat-square&label=Install%20MCP%20Server&color=24bfa5">](https://insiders.vscode.dev/redirect?url=vscode-insiders%3Amcp%2Finstall%3F%257B%2522name%2522%253A%2522lean-lsp%2522%252C%2522command%2522%253A%2522uvx%2522%252C%2522args%2522%253A%255B%2522lean-lsp-mcp%2522%255D%252C%2522env%2522%253A%257B%2522LEAN_PROJECT_PATH%2522%253A%2522path%2520to%2520lean%2520project%2520root%2522%257D%257D)


OR manually add config to `settings.json` (global):

```json
{
    "mcp": {
        "servers": {
            "lean-lsp": {
                "command": ["uvx"],
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


### Cursor

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

Lean LSP MCP currently provides various tools to interact with the Lean LSP server:

### Meta tools

- **lean_auto_proof_instructions**:
    Get detailed instructions on how to use the Lean LSP MCP to automatically prove theorems. This is a tool call because many clients do not support prompts yet.

### Core interactions

- **lean_diagnostic_messages**:
    Get all diagnostic messages for a Lean file. This includes infos, warnings and errors.

- **lean_goal**:
    Get the proof goal at a specific location in a Lean file. 

- **lean_term_goal**:
    Get the term goal at a specific position.

- **lean_hover_info**:
    Retrieve hover information for symbols, terms, and expressions in a Lean file.

- **lean_proofs_complete**:
    Check if all proofs in a file are complete.

### File operations

- **lean_file_contents**:
    Get the contents of a Lean file, optionally with line number annotations.

### Project-level tools

- **lean_project_path**:
    Get the path to the current Lean project root directory.

- **lean_project_functional**:
    Check if the Lean project and LSP server are functional and responding properly.

- **lean_lsp_restart**:
    Restart the LSP server and optionally rebuild the Lean project.


## Related Projects

- [LeanTool](https://github.com/GasStationManager/LeanTool): Provides an MCP "code interpreter" for Lean.


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
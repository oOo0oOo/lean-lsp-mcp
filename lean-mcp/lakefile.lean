import Lake
open Lake DSL

package LeanMcp where
  version := v!"0.1.0"
  keywords := #["mcp", "lsp", "lean"]
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

require Cli from git "https://github.com/leanprover/lean4-cli" @ "main"

lean_lib LeanMcp where
  globs := #[.one `LeanMcp, .submodules `LeanMcp]

@[default_target]
lean_exe leanmcp where
  root := `Main
  supportInterpreter := true

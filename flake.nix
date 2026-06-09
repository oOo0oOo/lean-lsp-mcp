{
  description = "Lean LSP MCP Server";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
    }:
    let
      inherit (nixpkgs) lib;

      # Read the workspace (pyproject.toml + uv.lock) — the single source of
      # truth for dependencies. Bumping a dep is `uv lock` + commit; the flake
      # follows automatically, no hand-pinned versions or hashes here.
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

      # Escape hatch for packages whose Nix build needs extra fixups. Empty for
      # now — add overrides here if a future dependency needs them.
      pyprojectOverrides = _final: _prev: { };
    in
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope
            (lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              pyprojectOverrides
            ]);

        venv = pythonSet.mkVirtualEnv "lean-lsp-mcp-env" workspace.deps.default;

        # mkApplication lives in pyproject-nix.build.util, which is a
        # `{ stdenv, python3 }`-style module — instantiate it via callPackage.
        inherit (pkgs.callPackage pyproject-nix.build.util { }) mkApplication;

        # Expose only this project's entry point (lean-lsp-mcp), not every
        # dependency's console script, so `nix profile install` stays clean.
        app = mkApplication {
          inherit venv;
          package = pythonSet."lean-lsp-mcp";
        };
      in
      {
        packages.default = app;

        apps.default = {
          type = "app";
          program = "${app}/bin/lean-lsp-mcp";
        };

        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
          ];
        };
      }
    );
}

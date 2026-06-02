{
  description = "Lean LSP MCP Server";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonEnv = pkgs.python3;

        leanclient = pythonEnv.pkgs.buildPythonPackage rec {
          pname = "leanclient";
          version = "0.10.0";
          src = pythonEnv.pkgs.fetchPypi {
            inherit pname version;
            sha256 = "sha256-VgppQRKXxIHsI6ZQZQJbS1VdzC/erJI5v9wrKSyLuyw=";
          };
          doCheck = false;
          pyproject = true;
          nativeBuildInputs = [ pythonEnv.pkgs.hatchling ];
          propagatedBuildInputs = with pythonEnv.pkgs; [
            orjson
            psutil
            tqdm
          ];
        };

        mcp = pythonEnv.pkgs.buildPythonPackage rec {
          pname = "mcp";
          version = "1.27.0";
          src = pythonEnv.pkgs.fetchPypi {
            inherit pname version;
            sha256 = "d3dc35a7eec0d458c1da4976a48f982097ddaab87e278c5511d5a4a56e852b83";
          };
          doCheck = false;
          pyproject = true;
          nativeBuildInputs = [ pythonEnv.pkgs.hatchling pythonEnv.pkgs.uv-dynamic-versioning ];
          propagatedBuildInputs = with pythonEnv.pkgs; [
            anyio
            httpx
            httpx-sse
            jsonschema
            pydantic
            pydantic-settings
            pyjwt
            python-multipart
            sse-starlette
            starlette
            typing-extensions
            click
            rich
            typer
            uvicorn
          ];
        };
      in
      {
        packages.default = pythonEnv.pkgs.buildPythonApplication {
          pname = "lean-lsp-mcp";
          version = "0.26.1";
          format = "pyproject";

          src = ./.;

          nativeBuildInputs = [
            pythonEnv.pkgs.setuptools
          ];

          propagatedBuildInputs = with pythonEnv.pkgs; [
            leanclient
            mcp
            orjson
            certifi
          ];

          # pyproject = true;
          meta = {
            description = "Lean Theorem Prover MCP";
            homepage = "https://github.com/oOo0oOo/lean-lsp-mcp";
            license = pkgs.lib.licenses.mit;
            mainProgram = "lean-lsp-mcp";
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.uv
          ];
        };
      }
    );
}

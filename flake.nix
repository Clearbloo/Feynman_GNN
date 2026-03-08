{
  description = "A very basic flake";

  inputs = { nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable"; };

  outputs = { self, nixpkgs }:
    let
      feynman_gnn = pkgs.python312Packages.buildPythonPackage {
        pname = "feynman_gnn";
        version = "0.1";
        src = ./.;
        format = "other";
      };
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      packages.${system}.default = feynman_gnn;
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          zed-editor
          vscodium

          (python312.withPackages (ps: [
            ps.ruff
            ps.pip
            ps.pytest
            ps.torch
            ps.pytorch-lightning
            ps.torch-geometric
            ps.pydantic
            ps.matplotlib
            ps.matplotlib-inline
            ps.numpy
            ps.networkx
            ps.pandas
            feynman_gnn
          ]))
        ];
      };
    };
}

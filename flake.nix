{
  description = "A very basic flake";

  outputs = {self, nixpkgs}:
  let
    pkgs = import nixpkgs {system="x86_64-linux"; config.allowUnfree=true;};
    rwkv = pkgs.python3Packages.buildPythonPackage rec {
      pname = "rwkvstic";
      version = "0.0.17";
      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "sha256-6Z/eqt1/f+BadH/OIrLUSe/v/mzjkCA9+tDz3vBVbDM=";
      };
      buildInputs = with pkgs.python3.pkgs; [hatchling];
      doCheck = false;
      format = "pyproject";
    };
  in {
    packages.x86_64-linux.hello = pkgs.hello;
    packages.x86_64-linux.default = self.packages.x86_64-linux.hello;

    devShell.x86_64-linux =
      with pkgs; mkShell {
        buildInputs = with python3.pkgs; [
          cudaPackages.cudatoolkit pytorch-bin inquirer scipy yarn rwkv tqdm
          transformers
        ];
      };
  };
}

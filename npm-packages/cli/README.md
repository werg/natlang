# `@natlang/cli`

The native natlang command line. It installs content addressed `.nlpkg` archives and launches their declared executable targets.

```sh
npm install --global @natlang/cli
natlang --help
natlang run path/to/program.nl
natlang app run path/to/application
natlang app list
natlang app run semantic-terminal
```

Source paths require no package installation. If a semantic run has no model
profile, the CLI lazily starts an owned local `llama-server` with the release
default model and stops it when the command exits.

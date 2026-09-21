# Development setup

From the repository root, prepare everything with:

```bash
scripts/setup_dev.sh
```

Use `scripts/setup_dev.sh --node-only` when Python development is unnecessary.
Both forms are safe to repeat. Node 22.13 or newer is required; full setup also
needs `uv` and Python 3.12.

Setup installs the pinned TypeScript dependencies and builds the Node and
browser outputs. Full setup also creates `.venv` and installs the Python project
with its development extras. It does not maintain an application catalog:
source applications and programs run directly from paths.

## Run from source

`scripts/natlang` is the checkout CLI. It rebuilds Node output whenever its
TypeScript inputs are newer than the compiled CLI.

Run a `.nl`, `.ts`, YAML, or JSON program directly:

```bash
scripts/natlang run path/to/main.nl
scripts/natlang run path/to/main.nl --inputs inputs.json --trace run.jsonl --json
```

Run an application by giving its manifest or a directory containing
`natlang.json`:

```bash
scripts/natlang app run path/to/application
scripts/natlang app run path/to/application/natlang.json
scripts/natlang app run packages/semantic-terminal.natlang.json
```

`scripts/natlang-app PATH` is a short spelling of `scripts/natlang app run
PATH`. There is no fixed or generated list of development applications. A
manifest beside its source normally resolves from its own directory. A
manifest kept separately, such as those in this repository's `packages/`
directory, searches parent directories for the source root. `--root DIR`
overrides that inference.

Arguments before `--` belong to natlang. Arguments after it belong to the
application:

```bash
scripts/natlang-app packages/evidence-console.natlang.json -- --documents evidence.json
scripts/natlang-app packages/notebook-console.natlang.json -- --notebook notebook.json
cat logs.jsonl | scripts/natlang-app packages/log-console.natlang.json --plain
```

Inspect a local application without installing it:

```bash
scripts/natlang app doctor packages/semantic-terminal.natlang.json --json
```

## Model lifecycle

No endpoint or model configuration is needed for the default local workflow.
The first semantic model turn:

1. selects the statically generated project default;
2. uses a verified checkout copy when present, otherwise downloads the verified
   GGUF once into the natlang cache;
3. starts `llama-server` on a private free loopback port;
4. waits for its health endpoint; and
5. terminates that owned process when the command closes or receives SIGINT or
   SIGTERM.

Crisp only programs never start or download a model. `llama-server` must be on
`PATH`; the official llama.cpp packages provide it (`brew install llama.cpp`,
`conda install -c conda-forge llama.cpp`, or another platform package). Point
at another binary with `NATLANG_LLAMA_SERVER`. `scripts/natlang doctor --json`
reports whether the managed local runtime is ready.

Useful local overrides are:

```bash
export NATLANG_MODEL_PATH=/path/to/model.gguf
export NATLANG_TEMPLATE=/path/to/chat-template.jinja
export NATLANG_LLAMA_SERVER=/path/to/llama-server
```

To use an already managed local or remote OpenAI compatible service instead,
configure both its endpoint and model. Natlang treats this process as externally
owned and never stops it:

```bash
export NATLANG_SERVER=http://127.0.0.1:8081
export NATLANG_MODEL=MODEL_ID
export NATLANG_API_KEY=KEY_IF_NEEDED
```

Persistent profiles live in `~/.config/natlang/config.json`:

```json
{
  "defaultProfile": "remote",
  "profiles": {
    "remote": {
      "endpoint": "https://example.invalid",
      "model": "MODEL_ID",
      "apiKeyEnv": "NATLANG_API_KEY"
    }
  }
}
```

The default is deliberately pinned, including its byte length and SHA-256
digest. `scripts/publish_browser_model.mjs` updates the browser catalog and
regenerates `ts-host/src/model-default.ts` in the same successful publication
flow. The checkout wrapper then rebuilds that static choice automatically on
its next invocation. An npm release snapshots the generated choice and its chat
template; installed users receive a newer default by updating natlang. Pass
`--download-url` during publication when that snapshot must be downloadable
outside the source checkout. Runtime startup never consults a mutable `latest`
URL.

## Installed applications and packages

Paths are the normal development and local authoring interface. Native packages
are optional distribution artifacts. Installed users can still discover and
run them:

```bash
natlang package install application.nlpkg
natlang app list
natlang app run application-name
natlang run package-name@1.0.0#target
```

See [Native packages and executables](NATIVE_PACKAGES.md) for archive and
distribution details.

## Builds and tests

The wrapper performs a fast Node build after TypeScript edits. Build the browser
bundle explicitly after browser, worker, bundler, or WASM changes:

```bash
npm --prefix ts-host run build
```

Run the TypeScript and Python suites with:

```bash
npm --prefix ts-host test
.venv/bin/python -m pytest -q
```

Checkout package objects, application state, caches, and traces live under
`.natlang/` and are ignored by Git.

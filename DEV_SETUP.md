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
with its development extras. It also installs `natlang` in `~/.local/bin` as a
symlink to this checkout's source wrapper. If that directory is missing from
`PATH`, setup prints the exact export to add to your shell profile.

This is the only repository setup command. It also checks an existing
`llama-server` version. If there is no compatible server, it asks before
downloading natlang's pinned, verified build into `.natlang/runtime`. The setup
does not modify a system installation. For unattended setup, `--yes` grants
permission for that managed download:

```bash
scripts/setup_dev.sh --yes
```

The install is safe to repeat and will not replace a `natlang` command owned by
another checkout or package installation. Use a different directory when you
want commands for multiple checkouts, or omit command installation:

```bash
scripts/setup_dev.sh --command-dir "$HOME/.local/dev-bin"
scripts/setup_dev.sh --no-command
```

To restore only the command without reinstalling dependencies, run
`scripts/setup_dev.sh --command-only`. `NATLANG_DEV_BIN_DIR` is the environment
equivalent of `--command-dir`. To disconnect this checkout, remove its symlink
from the selected command directory, for example with
`unlink "$HOME/.local/bin/natlang"` for the default. No shell alias is required.

## Run from source

The installed `natlang` command points at `scripts/natlang`. That checkout
wrapper rebuilds Node output whenever its TypeScript inputs are newer than the
compiled CLI, so source edits need no manual build step. It also keeps packages,
application state, caches, and traces under this checkout's `.natlang/` directory.
Commands can be invoked from any working directory; relative application and
program paths are resolved from that directory.

Run a `.nl`, `.ts`, YAML, or JSON program directly:

```bash
natlang path/to/main.nl
natlang path/to/main.nl --inputs inputs.json --trace run.jsonl --json
```

Run an application by giving its manifest or a directory containing
`natlang.json`:

```bash
natlang path/to/application
natlang path/to/application/natlang.json
natlang codebases/semantic_terminal
```

Discover runnable source applications without installing them:

```bash
natlang --apps
natlang --apps codebases
```

There is no fixed or generated development application list. Each runnable
application directory contains `natlang.json`. The CLI searches the manifest's
directory and its ancestors for the complete source root, since an application
may combine a natlang codebase with host adapters elsewhere in the repository.
`--root DIR` overrides that inference.

Arguments before `--` belong to natlang. Arguments after it belong to the
application:

```bash
natlang codebases/evidence_console
natlang codebases/notebook_console
natlang codebases/log_investigator
```

These open with useful starter state and describe possible next actions. Use
`/help` inside every interactive app. Evidence provides `/sources` and `/load
PATH`; notebook provides `/cells` and `/load FILE`; the log investigator
provides `/demo` and `/load FILE`; the semantic terminal provides `/recipes`.
The corresponding `--documents`, `--notebook`, and piped JSONL forms remain
available for automation and direct imports:

```bash
natlang codebases/evidence_console -- --documents evidence.json
natlang codebases/notebook_console -- --notebook notebook.json
cat logs.jsonl | natlang codebases/log_investigator --plain
```

Inspect a local application without installing it:

```bash
natlang --inspect codebases/semantic_terminal --json
```

## Model lifecycle

No endpoint, model, or system llama.cpp installation is needed for the default
local workflow. Setup discovers `NATLANG_LLAMA_SERVER`, a managed runtime, and
`llama-server` on `PATH`, in that order. Every candidate must report a version
inside natlang's tested range. Missing or incompatible ambient installations
are left untouched; with consent, natlang installs its verified build beside
them.

The first semantic model turn:

1. selects the statically generated project default;
2. uses a verified checkout copy when present, otherwise downloads the verified
   GGUF once into the natlang cache;
3. starts the selected `llama-server` on a private free loopback port;
4. waits for its health endpoint; and
5. terminates that owned process when the command closes or receives SIGINT or
   SIGTERM.

Crisp only programs never start or download a model. `natlang --setup` repeats
runtime discovery and installation, `natlang --runtime status --json` explains
every candidate, and `natlang --runtime install` explicitly installs the managed
build. Point at a custom compatible binary with `NATLANG_LLAMA_SERVER`.
`natlang --doctor --json` reports whether the complete local runtime is ready.

Useful local overrides are:

```bash
export NATLANG_MODEL_PATH=/path/to/model.gguf
export NATLANG_TEMPLATE=/path/to/chat-template.jinja
export NATLANG_LLAMA_SERVER=/path/to/llama-server
export NATLANG_RUNTIME_HOME=/path/to/natlang-runtimes
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

The llama.cpp runtime follows the same static publication rule. Maintainers run
`npm run runtime:update -- --version X.Y.Z` for a reviewed stable release. The
publisher follows that release's official build pointer, requires GitHub SHA-256
digests for every supported platform archive, and regenerates
`ts-host/src/llama-runtime-release.ts`. Installed CLIs never consult a mutable
latest release.

## Installed applications and packages

Paths are the normal development and local authoring interface. Native packages
are optional distribution artifacts. `natlang --apps` discovers source manifests;
`natlang --packages` reports archives installed into the content addressed package
store. Installed users can run a package by its name or exact target:

```bash
natlang --package install application.nlpkg
natlang --packages
natlang application-name
natlang package-name@1.0.0#target
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

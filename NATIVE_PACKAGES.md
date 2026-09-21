# Native natlang packages and executables

For ordinary work in this repository, start with
[DEV_SETUP.md](DEV_SETUP.md). Its wrappers run source paths directly. This
document describes the optional archive and distribution workflow.

Natlang applications can be shipped as deterministic `.nlpkg` archives. The
format packages natlang source trees, crisp functions, adapters, and binary
assets without adding module loading to the language core. A package may expose
ordinary source exports and executable terminal or command targets.

The distribution has three npm packages:

| Package | Contents |
|---|---|
| `@natlang/cli` | The small `natlang` executable |
| `@natlang/core` | Platform-neutral reduction, values, traces, agent, and crisp evaluator interface |
| `@natlang/node` | Node host, evaluator, terminal framework, model transport, and package APIs |
| `@natlang/browser` | Browser runtime, model client, worker, and WASM assets |

This split keeps the 35 MB unpacked browser inference bundle out of Node CLI
installs. The source checkout still builds all targets together so Node and
browser share the reduction implementation. `npm pack` in each directory under
`npm-packages/` creates the distribution tarball. Public registry publication
still requires the project owner to choose a license and supply credentials.

## Package an application

The checked schema is [package-manifest.schema.json](spec/package-manifest.schema.json).
Paths are relative to the package root. `include` is explicit; directories are
walked without following symbolic links. A target entry is a trusted native
adapter, while `reducer` and `view` name the natlang application logic.

```json
{
  "schema": "natlang.package/v1",
  "name": "@example/reviewer",
  "version": "1.0.0",
  "include": ["application", "program"],
  "targets": {
    "review": {
      "kind": "terminal",
      "entry": "application/target.mjs",
      "reducer": "program/reduce.nl",
      "view": "program/view.ts",
      "authority": ["filesystem:workspace"],
      "commands": ["git"]
    }
  },
  "engines": { "natlang": ">=0.1.0", "node": ">=22.13.0" }
}
```

Build, verify, install, inspect, and run it:

```bash
natlang package pack natlang.json --root . --out reviewer-1.0.0.nlpkg
natlang package verify reviewer-1.0.0.nlpkg --json
natlang package install reviewer-1.0.0.nlpkg
natlang package inspect @example/reviewer@1.0.0 --json
natlang doctor @example/reviewer@1.0.0#review
natlang run @example/reviewer@1.0.0#review --workspace . -- --application-option value
```

Packaging is unnecessary during authoring. Run a program, manifest, or
application directory directly:

```bash
natlang run path/to/main.nl
natlang app run path/to/application
natlang app run path/to/application/natlang.json
natlang app doctor path/to/application --json
```

Local manifests are validated with the same archive rules before launch, but
are not copied into the package store. The CLI infers the source root from the
manifest location and its ancestors; use `--root DIR` when the source lives
elsewhere.

For interactive use, discover applications and omit the exact version when it
is useful to follow the highest installed semantic version:

```bash
natlang app list
natlang app run reviewer --workspace . -- --application-option value
natlang app doctor reviewer --json
```

`app run` accepts the full package name or an unambiguous final name component.
It selects a sole target automatically; packages with several targets require
`--target`. Automation should continue to use the exact `run` form.

Install all archives in one command when packages depend on one another. The
installer validates the whole candidate set before publishing new objects:

```bash
natlang package install library-2.1.0.nlpkg app-1.0.0.nlpkg
```

Version 1 accepts exact versions, `*`, `latest`, caret, tilde, and `>=` ranges.
Resolution is offline: the command considers installed packages and archives
supplied by the caller. A registry client can later fetch candidates without
changing archive identity or runtime semantics.

## Archive and store guarantees

An archive is canonical JSON with a sorted file list. Every file is base64
encoded and records its byte length and SHA-256 digest, so text and binary
assets follow one rule. The archive digest covers the normalized manifest and
all file records. Verification rejects traversal, absolute paths, symbolic
links, duplicate paths, changed bytes, and noncanonical file order.

Installation writes an immutable content object, then atomically publishes a
`name@version` reference. The same version may be installed repeatedly only
with the same digest. There are no lifecycle scripts. Dependency constraints
are checked before a batch is installed. Each reference pins the selected
dependency versions and content digests; later installations cannot silently
change an existing application's graph. Cycles are rejected. JSON references
work across systems that do not support symbolic links.

`NATLANG_HOME`, `NATLANG_CONFIG_HOME`, `NATLANG_STATE_HOME`, and
`NATLANG_CACHE_HOME` override the platform package, configuration, application
state, and cache roots.

## Executable target contract

The entry module exports `createTarget(context)` unless `target.export` names a
different function. It returns an object with `run()` and optional `close()`.
The context contains immutable package identity, package and workspace roots,
the pinned dependency identities and roots, state and trace directories, target
arguments, terminal streams, an optional model driver, and the installed
runtime API.

Target adapters are trusted host code. The manifest’s `authority` and
`commands` fields make native access inspectable and let `natlang doctor` check
engines and executable dependencies. They are declarations, not a sandbox.
Natlang reducers own interpretation, planning, and decisions; adapters translate
typed events and expose exact native operations.

Without a profile, model use is lazy: the CLI starts an owned local
`llama-server` with the statically published default model on the first semantic turn
and closes it with the command. Crisp only targets do not start it. Model
profiles in the platform config directory select externally owned services:

```json
{
  "defaultProfile": "local",
  "profiles": {
    "local": {
      "endpoint": "http://127.0.0.1:8081",
      "model": "MODEL_ID",
      "apiKeyEnv": "NATLANG_API_KEY"
    }
  }
}
```

`NATLANG_SERVER`, `NATLANG_MODEL`, and `NATLANG_PROFILE` override the selected
profile. `NATLANG_MODEL_PATH`, `NATLANG_TEMPLATE`, and
`NATLANG_LLAMA_SERVER` customize managed local execution. Secrets remain in
environment variables. Packaging and terminal code do not impose model turn or
token caps.

## Included packages

The manifests in `packages/` build four complete applications:

- `@natlang/semantic-terminal`: semantic recipe selection and asynchronous job outcomes;
- `@natlang/evidence-console`: citation checked answers over local documents;
- `@natlang/log-console`: a JSONL event stream reduced into anomaly state;
- `@natlang/notebook-console`: semantic cell selection over SQLite and TypeScript cells.

Each application loads its reducer and view from its installed package object.
Adapters receive host classes from `@natlang/node` and have no repository
relative runtime import.

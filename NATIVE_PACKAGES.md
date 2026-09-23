# natlang packages and executables

For ordinary work in this repository, start with [DEV_SETUP.md](DEV_SETUP.md);
`natlang run` builds and runs source directories directly. This document covers
the npm distribution and the optional `.nlpkg` archive workflow.

## npm packages

| Package | Contents |
|---|---|
| `@natlang/cli` | The `natlang` executable |
| `@natlang/node` | Node runtime and compiler, terminal utilities, model session, package APIs |
| `@natlang/browser` | Browser runtime and in-page compiler, local model loader, WASM assets |

The browser package carries the 35 MB inference bundle, so it is separate from
Node installs. The checkout builds both from one implementation. `npm pack` in
each directory under `npm-packages/` stages and packs the build
(`scripts/stage-npm-package.mjs`). Public registry publication still requires
the project owner to choose a license and supply credentials.

## Package an application

An application is a TypeScript project with a `natlang.json` manifest (schema
[package-manifest.schema.json](spec/package-manifest.schema.json)). Paths are
relative to the package root; `include` is explicit. Each target names an entry
module and the exported function (default `main`) that receives the target
context:

```json
{
  "schema": "natlang.package/v2",
  "name": "@example/reviewer",
  "version": "1.0.0",
  "include": ["review.ts", "console.ts", "assess.nl", "tsconfig.json"],
  "targets": {
    "review": {
      "entry": "console.ts",
      "description": "Review a folder of notes.",
      "authority": ["filesystem:workspace"],
      "commands": ["git"]
    }
  },
  "engines": { "natlang": ">=0.1.0", "node": ">=22.13.0" }
}
```

```ts
import type { TargetContext } from '@natlang/node';
export async function main(context: TargetContext): Promise<number> {
  const report = await context.runtime.run(() => review(context.args[0]!));
  context.io.output.write(JSON.stringify(report) + '\n');
  return 0;
}
```

The context carries `args`, `io`, `workspace`, `stateDirectory`,
`traceDirectory`, the configured `model`, a `runtime` using it, and the package
and dependency identities. Build, verify, install, inspect, and run:

```bash
natlang package pack natlang.json --out reviewer-1.0.0.nlpkg
natlang package verify reviewer-1.0.0.nlpkg --json
natlang package install reviewer-1.0.0.nlpkg
natlang inspect @example/reviewer@1.0.0#review --json
natlang run @example/reviewer@1.0.0#review --workspace . -- --application-option value
```

Packaging is unnecessary during authoring: `natlang run path/to/application`
builds the project into `.natlang/build` and runs it against this CLI's runtime.
An installed package is built into its state directory on first run. Discover
applications and follow the highest installed version interactively:

```bash
natlang apps
natlang packages
natlang run reviewer -- --application-option value
```

`natlang run NAME` accepts the full package name or an unambiguous final name
component and selects a sole target automatically; use `--target` or the exact
`NAME@VERSION#TARGET` form in automation. Install archives that depend on each
other in one command; the installer validates the whole set first:

```bash
natlang package install library-2.1.0.nlpkg app-1.0.0.nlpkg
```

Dependency ranges accept exact versions, `*`, `latest`, caret, tilde, and `>=`.
Resolution is offline over installed packages and supplied archives.

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
state, and cache roots. `NATLANG_RUNTIME_HOME` overrides the managed native
runtime root.

## Authority and the model runtime

Entry modules are trusted application code. The manifest's `authority` and
`commands` fields make native access inspectable and let `natlang doctor` check
engines and executables; they are declarations, not a sandbox.

Without a profile, the CLI checks explicit, managed, and PATH llama.cpp
executables against its tested version range. If none is compatible, an
interactive run asks before installing the pinned official archive under the
user data directory; downloads are checked against a static byte length and
SHA-256 digest and never alter a system installation. `natlang setup` does this
ahead of time (`--yes` for unattended installs), and `natlang runtime status
--json` explains resolution. A run starts preparing the model immediately,
concurrently with building the application, and closes an owned server with the
command. Model profiles in the platform config directory select externally
owned services:

```json
{
  "defaultProfile": "local",
  "profiles": {
    "local": { "endpoint": "http://127.0.0.1:8081", "model": "MODEL_ID", "apiKeyEnv": "NATLANG_API_KEY" }
  }
}
```

`NATLANG_SERVER`, `NATLANG_MODEL`, and `NATLANG_PROFILE` override the selected
profile. `NATLANG_MODEL_PATH`, `NATLANG_TEMPLATE`, and `NATLANG_LLAMA_SERVER`
customize managed local execution. Secrets remain in environment variables.

## Included packages

Four applications in `applications/` ship as packages:

- `@natlang/evidence-console` (`applications/evidence/`): citation-checked answers over local documents;
- `@natlang/log-console` (`applications/logs/`): a JSONL event stream folded into incident state;
- `@natlang/notebook-console` (`applications/notebook/`): natlang-chosen cell execution over SQLite and JavaScript cells;
- `@natlang/semantic-terminal` (`applications/terminal/`): natural-language requests mapped to exact workspace recipes.

Each opens with useful starter state; `/help` lists its commands, and file flags
and stdin stay available for scripted use.

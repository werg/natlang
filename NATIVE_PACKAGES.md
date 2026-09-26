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
--json` explains resolution. A run starts an owned server when its first model
turn needs one and closes it with the command. Model profiles in the platform
config directory select a backend. For an externally owned service:

```json
{
  "defaultProfile": "remote",
  "profiles": {
    "remote": { "endpoint": "http://127.0.0.1:8081", "model": "MODEL_ID", "apiKeyEnv": "NATLANG_API_KEY" }
  }
}
```

The same profile resolution is used by `run`, `call`, `ask`, `setup`, and
`doctor`. Choose a profile with `--profile`, then `NATLANG_PROFILE`, then
`defaultProfile`. `--provider` or `NATLANG_PROVIDER` selects a Pi backend;
otherwise `NATLANG_SERVER` selects an OpenAI-compatible endpoint. `--model`
then `NATLANG_MODEL` override the profile's model ID. Provider selection takes
precedence over `NATLANG_SERVER`. Switching backend through an override drops
settings specific to the previous backend; shared `headers`, `apiKeyEnv`, and
`runtime` settings remain. A model ID without a provider or endpoint is invalid;
the managed local profile uses its pinned model. `NATLANG_MODEL_PATH`,
`NATLANG_TEMPLATE`, and `NATLANG_LLAMA_SERVER` customize managed local
execution. Secrets remain in environment variables.

| Profile field | Managed local | OpenAI-compatible endpoint | Pi provider |
| --- | --- | --- | --- |
| Selection | Neither `endpoint` nor `provider` | `endpoint`; optional `model` | `provider` and `model` |
| Credentials | None | `apiKeyEnv` (defaults to `NATLANG_API_KEY`) | Pi's provider environment variables, saved login, or `apiKeyEnv` |
| Request controls | `headers`, `request`, `local` | `headers`, `request` | `headers`, `piOptions`, `piPayload`, `piMode`, `modelOptions` |
| Natlang controls | `runtime` | `runtime` | `runtime` |

`request` adds fields to the OpenAI-compatible chat completion body. `local`
configures natlang's managed llama-server. `runtime` configures natlang's own
generation loop, independent of the backend. Pi's JSON fields cannot specify
JavaScript callbacks or SDK clients.

For a provider with a native API or a supported subscription, set `provider`
and `model` instead of `endpoint`:

```json
{
  "defaultProfile": "hosted",
  "profiles": {
    "hosted": { "provider": "anthropic", "model": "MODEL_ID" }
  }
}
```

`natlang models` lists Pi providers; `natlang models PROVIDER` lists their
model IDs. `natlang auth login PROVIDER` starts a supported OAuth sign-in,
and `natlang auth status` and `natlang auth logout PROVIDER` inspect or remove it.
API-key providers can use their usual environment variables, or set `apiKeyEnv`
on the profile. Natlang stores sign-in credentials under its platform config
directory and Pi refreshes expiring tokens. The Pi catalog in this version
supports subscription sign-in for `anthropic` (Claude Pro/Max), `openai-codex`
(ChatGPT Plus/Pro), `github-copilot`, `kimi-coding`, `xai`, and `meta`. Other
providers may offer OAuth without a subscription; `natlang auth login PROVIDER`
reports when a provider has no login flow. Availability of a particular model
depends on the provider account.

`natlang run`, `call`, `ask`, `setup`, and `doctor` also accept `--provider ID`
and `--model ID` for a single invocation.

Profiles can also set `piOptions` (Pi request options), `modelOptions` (catalog
model overrides), and `runtime` (natlang generation controls):

```json
{
  "profiles": {
    "reasoning": {
      "provider": "openai-codex",
      "model": "MODEL_ID",
      "piOptions": {
        "reasoningEffort": "high",
        "reasoningSummary": "auto",
        "cacheRetention": "long",
        "transport": "sse",
        "timeoutMs": 120000,
        "maxRetries": 1,
        "samplingParams": { "top_p": 0.9 }
      },
      "modelOptions": { "contextWindow": 128000, "maxTokens": 16384 },
      "runtime": {
        "temperature": 0,
        "turnTokens": 4096,
        "contextTokens": 120000,
        "maxTurns": 24,
        "seed": { "mode": "backend" }
      }
    }
  }
}
```

`piOptions` passes JSON request settings to Pi, including provider-specific
reasoning, cache, transport, metadata, regional and sampling settings. Its
`maxTokens` caps natlang's per-turn allowance. Natlang's configured temperature
overrides `piOptions.temperature`, and its seed overrides
`piOptions.samplingParams.seed` unless `runtime.seed.mode` is `backend`.
The OpenAI Codex Responses route rejects temperature, so natlang omits its
runtime temperature there.
`piOptions.env` supplies provider environment values for that profile. Top-level
`headers` take precedence over `piOptions.headers`. Per-request callbacks,
SDK `fetch` implementations, and direct API keys are unavailable in JSON
profiles; use the environment, `apiKeyEnv`, or the programmatic Pi backend.
Set `piMode` to `"simple"` to use Pi's provider-neutral `reasoning` and
`thinkingBudgets` options; the default `"native"` mode accepts each provider's
own options, such as `reasoningEffort` for OpenAI or `thinkingEnabled` for
Anthropic. Required-tool turns use the native mode in either case. Deferred
responses are not supported by natlang's synchronous turn contract.
`piPayload` shallowly merges JSON fields into Pi's formatted provider request
body for protocol experiments. It is applied after Pi builds the body, so fields
such as `messages` or `tools` can replace natlang's generated values if set.
`modelOptions` can override Pi catalog metadata such as `baseUrl`, `headers`,
`compat`, `samplingParams`, `contextWindow` and `maxTokens`; it cannot change the
selected model's `id`, `provider` or `api`. `natlang models PROVIDER --json`
shows the full catalog entry; `--refresh` refreshes dynamic catalogs.
For a model ID absent from Pi's catalog, set `modelOptions.inherit` to a listed
model from the same provider with the same API protocol. Natlang sends the new
`model` ID while using the template's protocol and capability defaults; override
its differing limits in `modelOptions`.
`runtime` accepts `maxTurns`, `maxTokens`, `turnTokens`, `temperature`,
`contextTokens` (or `null` to disable context compaction), `maxSeconds`,
`maxFailureRepairs`, `review`, and `seed`. Seed modes are `compatibility`,
`derived`, and `backend`; a numeric `seed.root` can be set when applicable.
These runtime settings apply to CLI `run`, `call`, and `ask` commands using that
profile. `runtime.review.driver` can only be supplied programmatically.

Managed local profiles can set `request` fields for llama.cpp chat completions,
plus a `local` object with `contextTokens`, `gpuLayers`, `parallel`,
`cacheRamMiB`, and extra llama-server `args`. The managed model path, loopback
host, port, and process lifecycle remain owned by natlang. The same `runtime`
controls apply. Endpoint profiles also accept `request` for raw chat-completion
body fields.

The Pi backend translates natlang's model turns and tool calls to provider
formats. Natlang still owns planning, guided tool selection, retries, execution,
and tracing. The managed local and endpoint backends retain direct access to raw
request fields and chat-completion wire behavior. The adapter maps natlang's
required-tool turns to Pi's native tool-choice option for APIs that provide one.
Sampling parameters, including seeds, only reach provider APIs that support
them. Profile JSON cannot carry callbacks or SDK client instances; those remain
programmatic integration points through `@natlang/node/model/pi`.

## Included packages

Four applications in `applications/` ship as packages:

- `@natlang/evidence-console` (`applications/evidence/`): citation-checked answers over local documents;
- `@natlang/log-console` (`applications/logs/`): a JSONL event stream folded into incident state;
- `@natlang/notebook-console` (`applications/notebook/`): natlang-chosen cell execution over SQLite and JavaScript cells;
- `@natlang/semantic-terminal` (`applications/terminal/`): natural-language requests mapped to exact workspace recipes.

Each opens with useful starter state; `/help` lists its commands, and file flags
and stdin stay available for scripted use.

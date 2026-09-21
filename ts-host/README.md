# TypeScript application host for natlang

`NatlangHost` is the Python-free TypeScript interpreter. `NativeNatlangHost` is an alias for it. Crisp TypeScript can run in a context that directly references application objects.

For agent-assisted development, install the [bundled authoring and integration
skills](../skills/README.md). They include portable references and a checked
multi-file example for both native hosts.

## Install and build

Applications install `@natlang/node`; the `natlang` executable is distributed
as `@natlang/cli`. Browser applications install `@natlang/browser`, keeping its
model worker and WASM assets out of Node CLI installations. The historical
`@natlang/typescript-host` name remains the monorepo build package. See
[native packages and executables](../NATIVE_PACKAGES.md).

Host implementers can use `@natlang/core`. Its `NativeRuntime` requires an
explicit `EvalEnvironment`, so the reduction layer does not choose Node VM,
browser eval, sandbox, filesystem, or process behavior. `@natlang/node`
exports a convenience `NativeRuntime` that supplies the Node evaluator.

From this repository:

```bash
cd ts-host
npm ci
npm run build
NATLANG_PYTHON=/path/to/natlang-python npm test
```

`NATLANG_PYTHON` must point to Python 3.11 or newer with the natlang Python dependencies installed for paired differential tests. The production package does not use Python. The package uses TypeScript's compiler API for transpilation and Node 22.13 or newer. It does not statically type-check arbitrary model-generated snippets; natlang checks values at the tree boundary. The package can be installed from this directory with `npm install /path/to/natlang/ts-host` after building it. The native conformance checks run with `npm run test:conformance` and do not require Python.

Import `NatlangHost` for runs. The declared conformance and paired trace coverage is recorded in [NATIVE_TYPESCRIPT_PORT.md](../plans/NATIVE_TYPESCRIPT_PORT.md).

For native CLIs and terminal dashboards, use the [terminal application
framework](TERMINAL_APPLICATIONS.md). It provides a queued natlang reducer/view
lifecycle, concurrent event sources, durable local sessions, structured terminal
views and a configurable model driver. The semantic terminal, log console,
evidence console and notebook console are executable integrations rather than
separate harnesses.

The native run request can include `review` with a reviewer driver, confidence threshold, action scope, and withdrawal policy. Reviews see proposed batches before any operation executes; a withdrawn batch can be retried once from the unchanged workspace. `validationFeedback` defaults to `caller`, matching the Python agent's behavior. The higher-level `BrowserNatlangApplication` and `TerminalNatlangApplication` default to `local`; for low-level runs use `local` to let the model repair missing results or rejected actions within its current episode.

## Run a program

```ts
import { NatlangHost, TypeScriptEnvironment, DesktopBindings } from '@natlang/node';

const desktop = new DesktopBindings();
const environment = new TypeScriptEnvironment({ mode: 'retained', host: desktop });
const host = new NatlangHost({ environment });

try {
  const result = await host.run({
    source: { kind: 'program', program: {
      $lambda: {
        type: 'Lambda<{ file: Text }, Num>',
        engine: 'typescript-host',
        code: 'return host.readText(args.file).length;',
      },
    } },
    inputs: { file: '/tmp/example.txt' },
    tracePath: '/tmp/example.trace.jsonl',
  });
  console.log(result.outcome, result.value);
} finally {
  host.close();
  desktop.close();
}
```

For a natural-language lambda, pass `modelTurn: async ({ messages, tools, temperature, seed, max_tokens }) => ...`. Return `{ calls: [[toolName, arguments], ...], text, completion_tokens }`; an empty `calls` array ends the episode. The callback receives the existing tools-v3 schema, including an explicit `engine` argument for `run_code`. You can instead pass `{ kind: 'definitions', entries, root }` or `{ kind: 'file', path }` as the source. The host loads `.nl`, `.ts`, YAML, and JSON sources. `options` accepts seed and model budgets. `streams: { over: asyncIterable }` binds a live root Fold input; the iterator's `next()` may await events without consuming model turns. `mapWorkers` requests parallel Map, but the shared engine serializes those calls unless the host is configured for safe parallel execution.

`capabilities: { 'service.operation': async (args) => value }` registers application callbacks for declared `fx` calls. The runtime enforces the lambda's `effects` list and records the request and outcome in its effect journal. A returned value must be portable JSON.

The native host runs Map slots serially by default. For independent pure work, pass `mapWorkers` and `parallelMapSafe: true`; parallel execution requires a fresh eval environment with no shared host object. Nested `NativeSourceWorkspace` invocations share the parent episode budget when the workspace is exposed directly on that host object.

The native host accepts declared capability callbacks that return values or promises. Authored crisp functions can `await` asynchronous application methods exposed through `host` and declared `fx` calls. `run_code` accepts an awaited expression. `NativeSourceWorkspace` provides versioned source description, type checking, and isolated child invocation through the shared host route.

## Eval environment and authority

`mode: 'fresh'` creates fresh TypeScript globals for each crisp eval. `mode: 'retained'` preserves globals across evals. In both modes, the supplied `host` object is passed by identity into the Node VM context. A retained environment can therefore share buffers, jobs, database clients, DOM-like objects, or application objects with authored crisp functions and `run_code` calls. `self`, `args`, and `locals` are frozen portable snapshots; they cannot modify natlang state directly. Results must be exact portable JSON values and are then checked against the destination's natlang type.

The shared engine is **trusted code**, not a sandbox. It can mutate exposed host objects before returning an invalid result or throwing. The VM's synchronous CPU timeout does not cancel a native operation or bound all memory use. Authored crisp functions and native `run_code` may await promises, but interruption cannot undo a native operation already in progress. Direct host access does not enter natlang's declared `fx` journal; declared `fx` calls do. The TypeScript host supports the `typescript-host` engine; the separate Python interpreter supports isolated QuickJS. Traces record the engine and host events but cannot reconstruct arbitrary native state or replay external effects.

An application host may expose `drainEvents()` returning portable observation
records. The runtime writes these as `host` trace events for successful and
failed crisp evals. They describe observed native operations; they do not make
those operations replayable or undo them.

## Browser host

For new web applications, use the [reusable browser client](BROWSER_CLIENT.md) to load local or hosted GGUFs and run natlang with one lifecycle API. It handles model templates, WebGPU selection, CPU fallback, asset URLs, model replacement, and per-run metrics. The lower-level host and model APIs below remain available for custom integrations.

The [interactive playground](playground/README.md) adds a browser-local multi-file editor, revisioned runs and trace inspection, reviewed cases, and a localhost training workbench. Start it with `npm run playground` after `npm ci`. Its case adapter sends exactly admitted natural-language leaf traces to the shared program IR and records explicit rejects for richer cases.

`@natlang/browser` exports `BrowserNatlangHost` and `BrowserLocalModel`. The browser bundle contains the same typed reducer, source parser, and model tool agent as the Node host. `BrowserLocalModel` runs a GGUF model locally through Wllama's browser worker; its weights can be cached in browser storage. No inference server or Python runtime is needed. Build with `npm run build:browser`; serve `dist/browser/natlang.js`, `wllama.wasm`, `wllama-compat.js`, and `wllama-compat.wasm` from the same directory. The compatibility assets keep Safari's GPU path self-hosted. Use HTTPS in deployment (localhost works for development) so WebGPU is available, and serve WASM as `application/wasm`. The compiled browser JavaScript is about 11 MB; the main and compatibility assets add about 23 MB, and model weights require additional browser storage and memory.

```ts
import { BrowserNatlangHost, BrowserLocalModel } from '@natlang/browser';

const application = { count: 2 };
const model = new BrowserLocalModel();
const template = await fetch('/models/templates/natlang-350M-v8-failures-pilot.jinja').then(r => r.text());
await model.loadFromUrl('/models/natlang-350M-v8-failures-pilot-Q8_0.gguf',
  { contextTokens: 8192, chatTemplate: template,
    onProgress: ({ loaded, total }) => console.log(loaded, total) });
const host = new BrowserNatlangHost({ host: application, model });
try {
  const result = await host.run({ source: { kind: 'program', program: {
    $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return the current count plus one.' },
  } }, options: { model: { max_turns: 12, turn_tokens: 512 } } });
  console.log(result.outcome, result.value, result.trace);
} finally {
  host.close();
  await model.close();
}
```

`model.loadFromUrl(url)` and `model.loadFiles([file])` also accept a hosted GGUF or a user-selected `File`; pass `wasmUrl`, `compatWorkerUrl`, and `compatWasmUrl` to the constructor when assets have different paths. By default the adapter probes the high-performance WebGPU adapter and requests **all model layers on GPU** when it supports `shader-f16`; otherwise it selects CPU. `model.diagnostics` reports the probe, selection reason, requested layers, and context, but Wllama does not expose the actual number of offloaded layers. Set `gpuLayers: 0` to request CPU, or a positive count when a model exceeds VRAM. The optional `firefoxGpuCompatibility: true` enables Wllama's slower Firefox compatibility path. You can still supply `modelTurn` per run to use another model backend. The host accepts `program`, checked `definitions`, and `files` sources. For `files`, pass `{ kind: 'files', root: 'tasks/main.nl', files: { 'tasks/main.nl': sourceText, ... } }`; the shared loader resolves companion files, `types.ts`, and `uses`.

The [browser pilot](examples/browser-local/index.html), playground, and browser board read the same `/models/browser-catalog.json` and select its published default. Training or checkpoint publication produces Q4_K_M by default for a smaller download and memory footprint; `Q8_0` can be published for comparison. Before the first publication, the built-in catalog points at the natlang v8 pilot Q4 and Q8 GGUFs and their official tool-call template. Those exact files are available as [private repository release assets](https://github.com/werg/natlang/releases/tag/natlang-v8-browser-pilot), rather than Git blobs. Run `gh release download natlang-v8-browser-pilot --dir models` from the repository root to install them. The browser pilot shows model size and storage headroom, automatically retries on CPU if GPU loading fails, allows a local file, and exports task traces and per-turn timing/token/schema metrics. By default the adapter compiles natlang's typed `write` and `call` alternatives into constrained model tools and compacts equivalent write destinations. Use `{ schemaMode: 'broad' }` to compare the former broad schema. The adapter enables prompt caching, serializes model turns, and retries one malformed tool call with corrective feedback. A context-full error asks for a smaller prompt or a larger loaded context.

Run `npm run test:browser` for a real Chromium interpreter smoke after installing the browser with `npx playwright-core install chromium`. Run `npm run test:browser -- --model --cpu --output=/tmp/browser-pilot.json` to exercise the local natlang GGUF on one inference task; add `--broad` to compare the broad tool schema. The pilot page can run three tasks under either schema mode and export both runs together. The live-model command is separate from the normal suite because inference is substantially slower. Set `NATLANG_CHROMIUM` to a Chromium executable when using an existing browser installation.

Use `--suite` for all three pilot tasks, or `--task=record` / `--task=nested-map` to run one structured case. Structured cases use a 4,096-token context by default in the CLI; `--context=8192` overrides it. `--gpu` launches Chromium with Linux Vulkan and NVIDIA f16 flags; `--cpu` requests no model layers on GPU, and `--quant=Q8_0` selects the larger checkpoint. `--probe=plain`, `--probe=tool`, and `--probe=natlang` isolate base generation, a simple tool, and the exact natlang request; `--probe-tokens=N` changes the probe limit. The model manifest pins each local v8 GGUF SHA-256 digest, while the page reports storage estimates and lets a user select a GGUF file.

For an interactive browser, run `npm run build` and then `npm run serve:browser -- --open-gpu` in `ts-host`. The server binds to localhost, serves COOP/COEP and byte ranges, and opens a separate Chromium profile with Vulkan and Dawn's NVIDIA f16 toggle on Linux. Use `npm run serve:browser` to print a URL for an existing browser. On macOS and Windows, `--open-gpu` leaves browser GPU selection to the platform. A normal Linux Chromium session needs the same launch flags and a full browser restart; the page cannot enable Dawn's f16 toggle itself. Check the page's GPU diagnostics and the pilot backend logs, not merely `navigator.gpu`, to establish offload.

The TypeScript agent follows the Python runtime's continuation checkpoints: after six durable work turns on an unfinished lambda, it asks for a short working note with no tools, stores the note on the lambda, and opens a fresh conversation from program and workspace state. Set `options.model.segment_turns` and `segment_messages` to tune rollover; their defaults are six turns and twelve messages. Set both to `null` to disable both triggers. The checkpoint is a conversation boundary, not an episode limit.

The local pilot server sends COOP/COEP headers and supports byte ranges for the GGUF; the page lets Wllama select its default WASM thread count when isolated. The adapter converts prior tool-call arguments from JSON strings into objects for the official LFM template. In a headless Q8 CPU trial, v8 loaded in 2.5 seconds and completed the leaf task in 72.7 seconds with two turns, 2,927 prompt tokens, 25 completion tokens, one valid action, and the correct value 7. With local validation feedback, the structured record task completed with an incorrect sum (5 instead of 8), and a nested Map completed with the input list unchanged ([2, 3] instead of [4, 6]). A Q4 GPU suite reproduced the same outcomes: 1/3 correct; its leaf, record, and nested Map tasks took 3.0, 8.4, and 7.4 seconds. These are model-quality failures on the pilot checkpoint. The Q4 leaf task completed correctly in 33.2 seconds on CPU and 2.7 seconds with WebGPU on this RTX 4060 host, both at 4,096 context tokens. Wllama's native logs reported 17/17 layers offloaded and a 216.41 MiB WebGPU model buffer. Timings are individual pilot runs under varying host load, not general speed guarantees. The browser API does not report the actual offload count; the pilot captures Wllama's native logs for that evidence.

Inference quality depends on the chosen model, available memory, and its tool calling support. More capable models need more storage and RAM. Multiple WASM CPU threads require cross origin isolation (COOP and COEP headers); WebGPU itself does not require those headers. The browser host returns the complete trace in memory; applications can save it with `new Blob([JSON.stringify(result.trace)], { type: 'application/json' })`. Browser capabilities use application callbacks, including DOM or network access when supplied by the application. Eval uses `Function` and direct `eval`, so the page's CSP must permit dynamic code execution. The shared eval environment is trusted application code, not an isolation boundary. Browser code does not expose Node process bindings, a disk trace writer, or a Node VM CPU timeout.

`DesktopBindings` supplies bounded text/byte file access and argv process execution, jobs, polling, cancellation requests, release, and event observations. Add application-specific objects to a separate host object as needed. Pass `observe: event => ...` to `TypeScriptEnvironment` to receive host and eval observations even without a trace file. `close()` on `NatlangHost` disposes its owned TypeScript context; close application-owned bindings separately. Aborting a run or hitting a timeout leaves external effect outcomes uncertain.

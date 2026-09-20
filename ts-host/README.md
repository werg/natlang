# TypeScript application host for natlang

`NatlangHost` is the Python-free TypeScript interpreter. `NativeNatlangHost` is an alias for it. Crisp TypeScript can run in a context that directly references application objects.

## Install and build

From this repository:

```bash
cd ts-host
npm ci
npm run build
NATLANG_PYTHON=/path/to/natlang-python npm test
```

`NATLANG_PYTHON` must point to Python 3.11 or newer with the natlang Python dependencies installed for paired differential tests. The production package does not use Python. The package uses TypeScript's compiler API for transpilation and Node 22.13 or newer. It does not statically type-check arbitrary model-generated snippets; natlang checks values at the tree boundary. The package can be installed from this directory with `npm install /path/to/natlang/ts-host` after building it. The native conformance checks run with `npm run test:conformance` and do not require Python.

Import `NatlangHost` for runs. The declared conformance and paired trace coverage is recorded in [NATIVE_TYPESCRIPT_PORT.md](../plans/NATIVE_TYPESCRIPT_PORT.md).

The native run request can include `review` with a reviewer driver, confidence threshold, action scope, and withdrawal policy. Reviews see proposed batches before any operation executes; a withdrawn batch can be retried once from the unchanged workspace. `validationFeedback` defaults to `caller`, matching the Python agent's behavior; use `local` to let the model repair missing results or rejected actions within its current episode.

## Run a program

```ts
import { NatlangHost, TypeScriptEnvironment, DesktopBindings } from '@natlang/typescript-host';

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

## Browser host

`@natlang/typescript-host/browser` exports `BrowserNatlangHost`, the same typed reducer and model tool agent bundled with a browser eval environment. Build it with `npm run build:browser`. The entry accepts in-memory `program` and checked `definitions` sources, input values, model callbacks, declared capabilities, and live root Fold streams. It returns the complete trace in memory.

```ts
import { BrowserNatlangHost } from '@natlang/typescript-host/browser';

const application = { count: 2 };
const host = new BrowserNatlangHost({ host: application });
const result = await host.run({ source: { kind: 'program', program: {
  $lambda: { type: 'Lambda<{}, Num>', code: 'return host.count + 1;' },
} } });
host.close();
```

The browser entry has no filesystem source loader, trace file writer, process bindings, or Node VM CPU timeout. File sources must be loaded by the application and supplied as in-memory programs or definitions. Eval uses `Function` and direct `eval`, so the page must allow dynamic code execution; it is trusted application code, not an isolation boundary. The portable natlang state, actions, reductions, and traces use the same implementation as the Node native host.

`DesktopBindings` supplies bounded text/byte file access and argv process execution, jobs, polling, cancellation requests, release, and event observations. Add application-specific objects to a separate host object as needed. Pass `observe: event => ...` to `TypeScriptEnvironment` to receive host and eval observations even without a trace file. `close()` on `NatlangHost` disposes its owned TypeScript context; close application-owned bindings separately. Aborting a run or hitting a timeout leaves external effect outcomes uncertain.

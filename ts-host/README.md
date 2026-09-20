# TypeScript application host for natlang

This package exposes two hosts. `NatlangHost` runs the complete Python interpreter through a bridge. `NativeNatlangHost` is a Python-free TypeScript interpreter under parity testing. Both can execute crisp TypeScript in a context that directly references application objects.

## Install and build

From this repository:

```bash
cd ts-host
npm ci
npm run build
NATLANG_PYTHON=/path/to/natlang-python npm test
```

`NATLANG_PYTHON` must point to Python 3.11 or newer with the natlang Python dependencies installed for the bridge tests. Use the project environment's `bin/python`. The package uses TypeScript's compiler API for transpilation and Node 22.13 or newer. It does not statically type-check arbitrary model-generated snippets; natlang checks values at the tree boundary. The package can be installed from this directory with `npm install /path/to/natlang/ts-host` after building it. The Python `natlang` package must be importable to the selected interpreter for `NatlangHost`. The native conformance checks run with `npm run test:conformance` and do not require Python.

For Python-free runs, import `NativeNatlangHost` instead. Its runtime is opt-in until the parity gate in [NATIVE_TYPESCRIPT_PORT.md](../plans/NATIVE_TYPESCRIPT_PORT.md) is complete.

## Run a program

```ts
import { NatlangHost, TypeScriptEnvironment, DesktopBindings } from '@natlang/typescript-host';

const desktop = new DesktopBindings();
const environment = new TypeScriptEnvironment({ mode: 'retained', host: desktop });
const host = new NatlangHost({ environment, python: '/path/to/.venv/bin/python' });

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

For a natural-language lambda, pass `modelTurn: async ({ messages, tools, temperature, seed, max_tokens }) => ...`. Return `{ calls: [[toolName, arguments], ...], text, completion_tokens }`; an empty `calls` array ends the episode. The callback receives the existing tools-v3 schema, including an explicit `engine` argument for `run_code`. You can instead pass `{ kind: 'definitions', entries, root }` or `{ kind: 'file', path }` as the source. File sources use the existing Python loader, including `.nl`, `.ts`, YAML and JSON programs. `options` accepts the Python `RunOptions` fields, including seed/model budgets. `streams: { over: asyncIterable }` binds a live root Fold input; the iterator's `next()` may await events without consuming model turns. `mapWorkers` requests parallel Map, but this host's shared engine is marked unsafe for parallel native access and the runtime serializes those calls.

`capabilities: { 'service.operation': async (args) => value }` registers application callbacks for declared `fx` calls in the isolated QuickJS engine. The runtime enforces the lambda's `effects` list and records the request and outcome in its effect journal. A returned value must be portable JSON.

The native host currently accepts synchronous declared capability callbacks. Authored crisp functions can `await` asynchronous application methods exposed through `host`. `NativeSourceWorkspace` provides versioned source description, type checking, and isolated child invocation through this route. `fx` calls in the current native eval engine are synchronous.

## Eval environment and authority

`mode: 'fresh'` creates fresh TypeScript globals for each crisp eval. `mode: 'retained'` preserves globals across evals. In both modes, the supplied `host` object is passed by identity into the Node VM context. A retained environment can therefore share buffers, jobs, database clients, DOM-like objects, or application objects with authored crisp functions and `run_code` calls. `self`, `args`, and `locals` are frozen portable snapshots; they cannot modify natlang state directly. Results must be exact portable JSON values and are then checked against the destination's natlang type.

The shared engine is **trusted code**, not a sandbox. It can mutate exposed host objects before returning an invalid result or throwing. The VM's synchronous CPU timeout does not cancel a native operation or bound all memory use. Returned promises are rejected; long work should use a retained host job and poll or a stream. Direct host access does not enter natlang's declared `fx` journal. Use the `quickjs-isolated` engine when declared `fx` capability enforcement and its journal are required. Traces record the engine and host events but cannot reconstruct arbitrary native state or replay external effects.

`DesktopBindings` supplies bounded text/byte file access and argv process execution, jobs, polling, cancellation requests, release, and event observations. Add application-specific objects to a separate host object as needed. Pass `observe: event => ...` to `TypeScriptEnvironment` to receive host and eval observations even without a trace file. `close()` on `NatlangHost` stops active interpreter processes and disposes the TypeScript context; close application-owned bindings separately. Aborting a run or hitting a timeout leaves external effect outcomes uncertain.

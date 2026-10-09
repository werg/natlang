# Hosts and model drivers

## One runtime, ordinary calls

| Target | Package | Context propagation |
|---|---|---|
| Node | `@natlang/node` | `AsyncLocalStorage` |
| Browser | `@natlang/browser` | Compiled code restores the task after each `await`; `runtime.bind` for uncompiled callbacks |

```ts
import { createNatlangRuntime, fileTraceSink, openAICompatibleModelTurn } from '@natlang/node';
import { handle } from './app.js';               // compiled with natlang build

const runtime = createNatlangRuntime({
  model: openAICompatibleModelTurn({ endpoint: 'http://127.0.0.1:8080', model: 'natlang' }),
  services: { wiki },                            // host capabilities, see below
  trace: fileTraceSink('.natlang/traces'),       // optional: one JSONL per invocation
});
const report = await runtime.run(() => handle(ticket));
```

`models: { small: driver }` names further models: a named function whose frontmatter says `model: small` runs on that one, and on `model` when no entry has its name.

`runtime.decide(fn, ...args)` runs a `readout: decision` function in a task of its own and returns `{ value, probabilities: [{ value, probability }], confidence, scored }`, so the host can act on probability floors; a driver without a scorer gives the sampled value with probability 1 and `scored: false`.

`runtime.run(fn, { services, signal, trace, name })` creates a task: the natlang calls made anywhere inside `fn` (including in libraries and callbacks) find it. Tasks run concurrently. A natlang call with no task fails with an error naming `runtime.run` and `runtime.bind`. Model options: `model` may be a driver function or `{ driver, maxTurns, maxTokens, turnTokens, temperature, maxFailureRepairs, … }` (all optional; none is set by default); `limits`, `seed`, `workspace`, `network`, `statistics`, and `progressJudge` are runtime options.

## Compile the application

`natlang build [PROJECT]` type-checks the project, plans every `nl` expression, checks callable folders, generates `foo.d.nl.ts` for each `.nl` file, and emits JavaScript. `natlang check` does the same without emitting. `buildProject({ project, runtimeModule })` and `checkProject` are the programmatic forms; `compileVirtualProject({ files }, runtimeNamespace)` compiles an in-memory project (browser pages, workers, tests). Uncompiled `nl` calls fail.

Named functions can also be loaded directly: `loadNatlang('review.nl')` (Node), `loadVirtualNatlang(files, 'review.nl')`, or `defineNatlang(nlSourceText)` for functions authored at run time (notebook cells, generated tools). `loadCallables('natlang.d')` loads a callable folder as a record. Tests that run application code importing `.nl` files under Vite or Vitest add `plugins: [natlangVitePlugin()]` (from `@natlang/node`): each `.nl` import loads as `loadNatlang` would, with the nearest `package.json` above it as the package root.

## Services

```ts
// app types (once):
declare module 'natlang:services' { export const wiki: WikiWorkspace; }
// callable-folder code:
import { wiki } from 'natlang:services';
```

Services are ordinary objects with methods, supplied per runtime or per task. Callable-folder TypeScript imports them from `natlang:services`; in eval they are named bindings. Every method call is traced as an effect. Services are read-only bindings: the model calls methods, it does not reassign properties.

Anything that stands for the world outside the program (stores, boards, simulators, verifiers, APIs) should be a service rather than callable-folder code: the model can read and edit callable-folder code, and it will edit a system that refuses it. Tell the model what a service is with its declaration, and limit who may use it where that matters:

```ts
const runtime = createNatlangRuntime({ model, services: { records, tables },
  // What the model is shown and read_code returns: a .d.ts body, wrapped as `declare namespace records { … }`.
  serviceDeclarations: { records: '/** Look a question up in the records. */\nexport function find(question: string): string;' },
  // tables is usable only in calls of answer/table_expert.nl and the calls they make; others see who can use it.
  serviceScopes: { tables: ['answer/table_expert.nl'] } });
```

Arguments reach a service as host-realm plain data, never as eval-realm objects, so a service may persist or compare them directly. A method that is an external effect to perform once (a provider request, a payment, a sent message) is listed under `ONCE_EFFECTS` (exported by `@natlang/node`): `{ [ONCE_EFFECTS]: ['send'], send(...) {...} }`. A repeated call with the same arguments on the same service object, from a re-run eval or another call, returns the earlier result; a failed call may run again. Hand a fresh service object to each unit of work (a task phase, a request) that may repeat the effect on purpose, and never list reads.

Both options also exist per task (`runtime.run(fn, { services, serviceDeclarations, serviceScopes })`). A service without a declaration is listed by its method names only. Scopes name functions by their source path relative to the loaded program. Importable packages need no declaration: `read_code("pkg")` reads a package's exports from its own type declarations.

## Model transport

A driver receives `{ messages, tools, temperature, seed, max_tokens }` and returns ordered tool calls plus optional text and usage:

```ts
{ calls: [['eval', { code: 'return await helper(sample)' }]], text: '', completion_tokens: 42, prompt_tokens: 700 }
// a later turn with no calls, e.g. { text: 'done' }, returns the staged value
```

A driver is an ordinary function, so it composes: wrap one to limit requests in flight, to route some calls elsewhere, or to answer a known call itself (the first pi app answered its codemode call with an `eval` of the agent's script); copy the wrapped driver's properties (`Object.assign(wrapper, driver)`) so its decision scorer stays reachable. `openAICompatibleModelTurn({ endpoint, model, concurrency })` caps requests in flight over its turns and its decision scorer together (pass one `requestLimit(n)` to several drivers to share a cap); profiles take `concurrency` too, or `NATLANG_CONCURRENCY` per process. Requests wait in one model scheduler per backend (`createScheduler`, `model/scheduler.ts`): it sends up to `maxConcurrent` at once, releases requests that are ready together as one batch, serves turns of calls already running before new calls, sends requests with the same system prompt next to each other, and cancels queued requests by signal. Pass it as `concurrency` (a bare number keeps the plain FIFO `requestLimit`). Profile knobs: `batching: { mode: 'server-continuous' | 'explicit-batch' | 'serial', maxConcurrent, coalesceMs, priority, scoreEndpoint }` (a managed local server defaults `maxConcurrent` to its slot count; `NATLANG_CONCURRENCY` still overrides `concurrency`), and for a managed local server `local: { parallel, contextTokens, memoryBudgetMiB, kvBytesPerToken }`: unset `parallel` is sized from memory (`NATLANG_MODEL_MEMORY_BUDGET_MIB`, `NATLANG_MODEL_KV_BYTES_PER_TOKEN`; formula in `model/server-slots.ts`), `contextTokens` is per slot, and requests carry `cache_prompt: true`. `natlang doctor` reports `serverSlots` and `batching`. Decision scorers expose `scoreMany(items)`; decision readouts that arrive together are scored together, and with `mode: 'explicit-batch'` plus `scoreEndpoint` through one prefix-plus-continuations request (plans/BATCHED_EXECUTION.md). Each model request's trace event records `batch_id`, `batch_size`, `in_flight` and `queue_wait_ms`; `natlang traces occupancy` summarises them. `natlang check` warns about a `for…of` that awaits one independent nl call per item (`nl-sequential-loop`) and suggests `Promise.all(items.map(...))`; loops are never parallelized silently. `openAICompatibleModelTurn` adapts an OpenAI-compatible server (aliases, retries of malformed tool calls, raw exchanges via `onExchange`). `createManagedModelSession` and `natlang setup` run the managed local llama.cpp runtime. Keep provider quirks in the driver, not in `.nl` source. If `max_tokens` is null, omit a provider field that requires an integer.

## Folders

A directory reducer's first parameter is a `Folder` (or a `FolderHandle` from `folder.dir(path)`). Node's `openFolder(dir)` fronts a directory lazily; `saveFolder(dir, folder)` writes its committed changes back, each file atomically. A direct reducer call returns its typed value and discards file changes; `folder.apply(reducer, ...args)` keeps the committed ones.

A reducer call is a transaction: open the folder with `'overlay'` access (writes stay in memory), `apply` the reducer, and commit `await folder.diff()` yourself. `saveFolder` replaces files one by one, so for all-or-nothing durability write a log record with every changed file's new content first, then the files, and finish an interrupted commit from the last record on open (`applications/nldb`). Serialize writers; a read-only question is a direct call on an overlay, so anything it writes is discarded.

## Authority

Services, live values, and packages available to eval run with the application's authority. `workspace` selects the `package.json` whose dependencies eval and callable-folder code may import; `network` allows `fetch` in eval. A timeout or failed validation cannot undo an effect that already happened.

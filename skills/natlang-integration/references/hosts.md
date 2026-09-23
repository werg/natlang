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

`runtime.run(fn, { services, signal, trace, name })` creates a task: the natlang calls made anywhere inside `fn` (including in libraries and callbacks) find it. Tasks run concurrently. A natlang call with no task fails with an error naming `runtime.run` and `runtime.bind`. Model options: `model` may be a driver function or `{ driver, maxTurns, maxTokens, turnTokens, temperature, validationFeedback, … }`; `limits`, `seed`, `workspace`, `network`, `statistics`, and `progressJudge` are runtime options.

## Compile the application

`natlang build [PROJECT]` type-checks the project, plans every `nl` expression, checks callable folders, generates `foo.d.nl.ts` for each `.nl` file, and emits JavaScript. `natlang check` does the same without emitting. `buildProject({ project, runtimeModule })` and `checkProject` are the programmatic forms; `compileVirtualProject({ files }, runtimeNamespace)` compiles an in-memory project (browser pages, workers, tests). Uncompiled `nl` calls fail.

Named functions can also be loaded directly: `loadNatlang('review.nl')` (Node), `loadVirtualNatlang(files, 'review.nl')`, or `defineNatlang(nlSourceText)` for functions authored at run time (notebook cells, generated tools). `loadCallables('natlang.d')` loads a callable folder as a record.

## Services

```ts
// app types (once):
declare module 'natlang:services' { export const wiki: WikiWorkspace; }
// callable-folder code:
import { wiki } from 'natlang:services';
```

Services are ordinary objects with methods, supplied per runtime or per task. Callable-folder TypeScript imports them from `natlang:services`; in eval they are named bindings, shown to the model with their types. Every method call is traced as an effect. Services are read-only bindings: the model calls methods, it does not reassign properties.

## Model transport

A driver receives `{ messages, tools, temperature, seed, max_tokens }` and returns ordered tool calls plus optional text and usage:

```ts
{ calls: [['eval', { code: 'result = await helper(sample)' }], ['mark_lines', { start: 1, end: 2 }]],
  text: '', completion_tokens: 42, prompt_tokens: 700 }
```

`openAICompatibleModelTurn` adapts an OpenAI-compatible server (aliases, retries of malformed tool calls, raw exchanges via `onExchange`). `createManagedModelSession` and `natlang setup` run the managed local llama.cpp runtime. Keep provider quirks in the driver, not in `.nl` source. If `max_tokens` is null, omit a provider field that requires an integer.

## Folders

A directory reducer's first parameter is a `Folder` (or a `FolderHandle` from `folder.dir(path)`). Node's `openFolder(dir)` fronts a directory lazily; `saveFolder(dir, folder)` writes its committed changes back atomically. A direct reducer call returns its typed value and discards file changes; `folder.apply(reducer, ...args)` keeps the committed ones.

## Authority

Services, live values, and packages available to eval run with the application's authority. `workspace` selects the `package.json` whose dependencies eval and callable-folder code may import; `network` allows `fetch` in eval. A timeout or failed validation cannot undo an effect that already happened.

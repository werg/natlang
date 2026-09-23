# Browser applications

## Model and runtime

```ts
import * as natlang from '@natlang/browser';

const catalog = await natlang.loadBrowserModelCatalog();
const entry = catalog.models.find(model => model.id === catalog.defaultId)!;
const { model, status } = await natlang.loadBrowserLocalModel(
  { kind: 'url', id: entry.id, url: entry.url, templateUrl: entry.templateUrl },
  { contextTokens: entry.contextTokens });
const runtime = natlang.createNatlangRuntime({ model: model.turn, services: { board } });
```

`loadBrowserLocalModel` fetches a separately published chat template, runs single-threaded when the page is not cross-origin isolated, and falls back to CPU when an automatic GPU load fails; `status.gpuFallbackReason` reports it. Requested GPU offload is not proof of GPU residency. Do not hard-code a historical checkpoint; use the published catalog or an explicit GGUF with its template.

## Application code

Compile the application's TypeScript with the in-page compiler, or ahead of time with `natlang build --target browser`:

```ts
const compiled = natlang.compileVirtualProject({ files: { 'board.ts': source } }, natlang);
if (!compiled.ok) throw new Error(natlang.formatDiagnostics(compiled.diagnostics));
const board = compiled.require('board.ts');

const loop = new natlang.EventLoop({ initialState: board.initialBoard(), reduce: board.reduce, view: board.view,
  step: (fn, context) => runtime.run(fn, { signal: context.signal }),
  onCommit: commit => journal.persist(commit), onTransition: transition => renderer.render(transition.view) });
const renderer = new natlang.BrowserDomRenderer(root, event => loop.dispatch(event));
await loop.start();
```

`EventLoop` applies events one at a time in arrival order, suppresses duplicate event IDs, awaits `onCommit` before the view, and retries a failed view with `refresh()` without replaying the event. `cancel()` aborts the active step. Durable exactly-once behavior still needs your persistent journal; pass the recovered state, `initialRevision`, and `seenEventIds` on restart.

Compiled browser code restores the natlang task across `await`. Callbacks invoked by uncompiled code (DOM listeners, timers, third-party libraries) must be wrapped with `runtime.bind(fn)` or call into `runtime.run`.

## Give natlang an active UI role

- Natlang reducer, exact presentation: semantic commands with predictable layout.
- Natlang view plan, exact layout: natlang chooses grouping, explanation, and suggestions; code builds and checks the tree (the task board example).
- Generated interfaces: natlang writes a view or module whose controls emit events to real handlers (the Studio research lab's generated modules).

`BrowserDomRenderer` renders checked `UiNode` trees with text nodes and typed events; it rejects executable markup. Keep high-frequency input and animation exact.

## Playground projects and workers

`newPlaygroundProject`, `runPlaygroundProject`, `projectEntry`, `projectSignature`, and `traceFrame` run and inspect small projects (a `.nl` root called with named inputs, or a `.ts` module exporting `main(inputs)`). Run user programs in a worker when they may block the page; post model turns back to the page's model, as Studio's `child-worker.mjs` does.

## Deployment

Serve `natlang.js`, `wllama.wasm`, `wllama-compat.js`, and `wllama-compat.wasm` from the build with the correct MIME types, and model assets with range support. Browser GPU access needs localhost or HTTPS; COOP/COEP enables multithreaded WASM. Eval is trusted page code and needs the corresponding CSP permissions.

Anchors: `ts-host/examples/browser-board/`, `ts-host/examples/browser-local/`, `ts-host/playground/`, `ts-host/studio/`, `ts-host/test/browser.test.mjs`, `ts-host/scripts/browser-pilot.mjs`.

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

`model.turn` carries `decide`, so decision readouts (`runtime.decide`, `decisionReadout`) are scored in the page. The runtime's `guidance` option (guided generation: tool-call names, and eval code checked line by line for repetition, redeclaration and TypeScript syntax, with rollback) also works with the in-page model; it helps small models most. For a tuned release, `natlang.loadBrowserRelease(releaseUrl, { baseGgufSha256 })` returns `adapters`, `nativeStates` (the soft instruction and skill states) and `weights`; pass the first two to `loadBrowserLocalModel` as `loraAdapters` and `nativeStates`, and `weights` to `createNatlangRuntime`. A release exported for another base GGUF runs untuned, and its soft states run degraded.

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

`context.after(work)` runs slow work (a model call, a request) once the step's state is committed and dispatches the event it returns, so the queue is not held while it runs; `close()` aborts it. `context.now` is the step's time, stored with its commit (`clock` replaces `Date.now`). `wakeAt(state)` keeps one timer that dispatches `{ kind: 'wake', at }`: a deadline or reminder is state, re-armed from restored state after a restart; each time fires once. `KeyedEventLoop` runs one loop per key (a user, an order, a document): events of one key apply in order, keys in parallel, with per-key `restore` and `onCommit(key, commit)`. `applications/workflow` (`WorkflowDesk`) decides orders side by side and reconciles an unacknowledged charge with `wakeAt`. `applications/helpdesk` is a multi-user service on the same pieces: one loop per ticket, triage and a draft reply as follow-up work, response deadlines as state, and an HTTP interface.

Keyed applications need three things the library leaves to them:
- **Resume deadlines.** A key's loop starts with its next event, so at startup open the loops of stored keys that have a wake time (`loops.loop(key)`).
- **Store applied event IDs.** `onCommit` runs before the event joins `loop.seenEventIds`, so store `[...loop.seenEventIds, commit.event.id]` together with the state.
- **Keep the key in state.** A wake event carries no key, and a follow-up result can arrive after newer input. Tag such results with what they read (a message count, a revision) and apply only newer ones.

Compiled browser code restores the natlang task across `await`. Callbacks invoked by uncompiled code (DOM listeners, timers, third-party libraries) must be wrapped with `runtime.bind(fn)` or call into `runtime.run`.

## Give natlang an active UI role

- Natlang reducer, exact presentation: semantic commands with predictable layout.
- Natlang view plan, exact layout: natlang chooses grouping, explanation, and suggestions; code builds and checks the tree (the task board example).
- Generated interfaces: natlang writes a view or module whose controls emit events to real handlers (the Studio research lab's generated modules).

`BrowserDomRenderer` renders checked `UiNode` trees with text nodes and typed events; it rejects executable markup. Keep high-frequency input and animation exact.

## Mount the dashboard pages

The dashboard's pages (`ts-host/dashboard/pages/*.js`: overview, programs, runs, traces, playground, models, system) are
framework-free modules a host page can mount, so a panel shows natlang's live execution view, typed run forms and traces
instead of reimplementing them. A page is `mount(el, host) => dispose`; `host = { api, runs?, navigate, toast, arg?, href?, refreshStatus?, h? }`
(types in `ts-host/dashboard/types.d.ts`). `api` is a `DashboardApi` (groups `status`, `models`, `programs`, `traces`, `system`, `jobs`;
`httpDashboardApi({ baseUrl, token })` talks to `natlang dashboard`); `runs` is a `RunHost` (`kinds`, `list`, `start`, `watch`, `subscribe`,
`stop`, `remove`, `input`), either `httpRunHost(api)` or `inProcessRunHost(runtime, executors)`, which runs each spec inside `runtime.run`
with `onLive` and prints the executor's result to the run's output. Supply only the groups and run kinds the host has:
`availablePages({ api, runs })` returns the pages those support (no `models` group, no Models page; no `ask` kind, no Playground), and a
page mounted without what it needs throws. `navigate(route)` takes `page` or `page-ARG` and must mount that page again even when it is
current; dispose the returned function when the host leaves the page. Load `style.css` once: its rules apply only under `.nl-dash`
(a mounted element gets the class when no ancestor has it) in the `natlang` cascade layer, so host rules win and nothing leaks out.
Vibestudio vendors the modules with `node scripts/stage-vibestudio.mjs --out DIR --only dashboard`. Anchors: `ts-host/scripts/dashboard-embed-smoke.mjs`,
`ts-host/test/dashboard-run-host.test.mjs`.

## Playground projects and workers

`newPlaygroundProject`, `runPlaygroundProject`, `projectEntry`, `projectSignature`, and `traceFrame` run and inspect small projects (a `.nl` root called with named inputs, or a `.ts` module exporting `main(inputs)`). Run user programs in a worker when they may block the page; post model turns back to the page's model, as Studio's `child-worker.mjs` does.

## Deployment

Serve `natlang.js`, `wllama.wasm`, `wllama-compat.js`, and `wllama-compat.wasm` from the build with the correct MIME types, and model assets with range support. Browser GPU access needs localhost or HTTPS; COOP/COEP enables multithreaded WASM. Eval is trusted page code and needs the corresponding CSP permissions.

Anchors: `ts-host/examples/browser-board/`, `ts-host/examples/browser-local/`, `ts-host/playground/`, `ts-host/studio/`, `ts-host/test/browser.test.mjs`, `ts-host/scripts/browser-pilot.mjs`.

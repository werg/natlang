# Native natlang frontends

## Model and application lifecycle

Use `BrowserNatlangClient` once per owned model lifecycle. Load the current published catalog via `loadDefaultModel()` or specify an explicit GGUF and matching tool template. Do not hard-code a historical checkpoint merely because a README example names it. Browser weights may need to be supplied separately from source checkout/package.

```ts
import { BrowserNatlangClient, BrowserNatlangApplication } from '@natlang/browser';

// files contains types.ts, reduce.nl, view.nl and their lexical helpers.
export async function mountApplication({ files, initialState, revision, journal, renderer, host }) {
  const client = new BrowserNatlangClient({ host, mode: 'retained' });
  let app;
  try {
    await client.loadDefaultModel();
    app = new BrowserNatlangApplication({
      client, source: { files, reducer: 'reduce.nl', view: 'view.nl' },
      initialState, initialRevision: revision, seedRoot: 17,
      onCommit: async commit => journal.persist(commit),
      onTransition: transition => renderer.render(transition.view),
      onFailure: failure => renderer.showFailure(failure.detail),
    });
    await app.start();
    return {
      dispatch: event => app.dispatch(event),
      refresh: () => app.refresh(),
      close: async () => { await app.close(); await client.close(); },
    };
  } catch (error) {
    if (app) await app.close();
    await client.close();
    throw error;
  }
}
```

`journal` and `renderer` are application interfaces to implement, not natlang exports. A supplied `modelTurn` permits another backend or a clearly labelled fixture. The reducer signature is `(state: State, event: Event) -> State`; the view takes `state` and returns the chosen UI type. `BrowserAppEvent` has `id`, `kind`, and optional Text `value`; encode richer payloads deliberately or use a typed application layer.

`dispatch` queues events. Stable IDs suppress duplicate events for the current application instance; durable exactly-once behavior needs a persistent event/operation journal. Supply recovered `initialRevision` as well as state. `onCommit` is awaited before view computation. `refresh()` retries presentation without rerunning the reducer. Persist traces intentionally rather than accumulating them forever in memory. Failed reductions can still have host effects; a preserved previous State does not imply rollback.

`cancel()` aborts the active application run. Incoming events normally wait; they do not mutate a running lambda. One client rejects simultaneous runs/model replacement while busy. Separate clients and isolated environments are needed for genuinely independent ownership.

## Give natlang an active UI role

Choose per use case:

- Natlang reducer, crisp presentation: semantic commands and workflows with predictable layout.
- Natlang reducer and view plan: the model chooses grouping, explanation, interaction, and what evidence to show; a renderer implements the layout.
- Natlang-generated trees or modules: create a task-specific interface, bind controls to real natlang handlers, inspect events, revise the interface as state changes.

The shared `BrowserDomRenderer` accepts checked DOM descriptions and emits events; it is not a raw HTML renderer. React/canvas/custom renderers can consume other typed view values. Use crisp high-frequency input handling where inference per keystroke or animation frame would be wasteful. Natlang can still decide what the controls mean and what operation follows.

Inquiry Lab's `ts-host/studio/shared/generated-module.mjs` is a richer application library: `{module:{html,style,script,title},bindings}`; scripts emit declared controls through `natlang.emit`, and drafts can use `natlang.draft` / `natlang.drafts`. Its presentation runs in an origin-isolated iframe without network, with handlers pinned to a source manifest. This is not automatically a facility of every `BrowserNatlangClient`. Reuse or extract that library explicitly and test event bindings, native data access, persistence, and version pinning. Generated JavaScript should not be the only place domain conclusions live.

## Deployment and resource ownership

Build and serve `natlang.js`, `wllama.wasm`, `wllama-compat.js`, and `wllama-compat.wasm` from the runtime's build. Configure explicit worker URLs if bundling changes their locations. Serve WASM with its correct MIME type, model assets with suitable download/range support, and the matching template. Localhost or HTTPS is needed for browser GPU access. COOP/COEP enables multithreaded WASM CPU execution; WebGPU itself does not require isolation headers. Test the actual browser and asset paths.

The client reports GPU selection and fallback diagnostics; requested offload is not proof of actual GPU residency. Inspect measurements/logs when claiming acceleration. A published model's context must fit its memory/backend configuration. Do not start additional model servers or reconfigure a shared GPU to validate UI wiring; use scripted fixtures first and an explicit live-model scenario separately.

Eval is trusted page code and needs the applicable CSP permissions. Shared DOM/host objects provide real authority. A fresh eval context does not isolate those objects. Generated presentation isolation and evaluator isolation are separate design choices.

Repository anchors: `ts-host/BROWSER_CLIENT.md`, `ts-host/FRONTEND_APPLICATIONS.md`, `ts-host/src/browser/application.ts`, `ts-host/src/browser/client.ts`, `ts-host/examples/browser-board/`, and `ts-host/studio/research/`.

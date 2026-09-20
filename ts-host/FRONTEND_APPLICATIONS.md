# Natlang-driven browser applications

`BrowserNatlangClient` loads a local model and runs natlang. The new
`BrowserNatlangApplication` uses that API to run a typed reducer and a view
function for each UI event. It is a library for browser applications, not a
new language primitive or a universal component framework.

```ts
const app = new BrowserNatlangApplication({
  client,
  source: { files, reducer: 'board/reduce.nl', view: 'board/view.nl' },
  initialState,
  seedRoot: 17,
  onTransition: ({ view }) => renderer.render(view),
});
await app.start();
// Browser controls, WebSocket messages and timers all enter the same queue:
await app.dispatch({ id: 'event-1', kind: 'command', value: 'Add a task' });
```

The application queue gives one state owner an explicit event order. Each
completed reducer invocation is one Fold-like step. The new state commits only
when its natlang run completes with a typed value. A view is then computed
from that state. Events arriving during a model turn wait in the queue; they
do not interrupt or rewrite the active lambda. Failures preserve the last
committed state. An observer receives each run and its trace, then decides
whether to persist it. The wrapper does not retain full traces indefinitely.

The current browser host can consume an open stream Fold, but `run()` returns
its result when the stream closes. That shape is useful for batch reduction;
it does not give a frontend a view after every incoming event. The per-event
application wrapper supplies that presentation boundary without changing Fold
or adding ambient interrupts. An eventual host step callback could let one
open Fold stream publish intermediate snapshots, if a real application needs
it. The event queue is the simpler first contract.

## UI choices

1. **Natlang reducer, crisp view.** Use for high-rate controls and exact
   layout. Natlang still owns application state transitions.
2. **Natlang reducer and view plan, crisp renderer.** A natlang function
   chooses grouping, wording and visible actions. A crisp helper checks IDs,
   data coverage and control wiring, then produces DOM data. The
   [browser task board](examples/browser-board/) demonstrates this option.
3. **Natlang-generated view tree.** The optional `BrowserDomRenderer` accepts
   a small tree of permitted tags and emits typed events from controls. It
   creates nodes with `textContent`; raw HTML and JavaScript attributes are
   outside the view type. This is useful for experiments, though generating
   a large tree on every keystroke would be slow and difficult for a small
   interpreter model.

Application-specific view schemas can be passed to another renderer such as
React or a canvas host. The client and application queue have no dependency
on the DOM renderer. Native DOM, model, editor and media objects may live in
the selected browser eval environment; portable state and view descriptions
cross the natlang boundary. A page-authored programme should get a separate
host environment with only its intended capabilities.

## Operational details

- Give each event a stable ID. The application suppresses duplicate IDs for
  its lifetime; a durable application needs a host-owned event journal and
  recovery contract. The queue records arrival order through reducer runs.
- Supply a root seed to derive an invocation seed from the event ID and state
  revision. Reproducing a model decision also requires the same checkpoint,
  source, prompt context, engine bindings and ordered events.
- A `retained` browser client now keeps one eval environment across its runs.
  This retains explicitly shared browser host objects while the client lives.
  App state still travels as a typed reducer value, so it can be inspected
  and recovered independently of native objects.
- Render controls through the optional DOM adapter or an application renderer.
  A button can read a named input and emit one typed command; inputs can also
  emit `change`. Both avoid a model call on every keystroke. Frequent
  pointer/motion/scroll events should stay in crisp
  browser code and become occasional semantic events at decision boundaries.
- Keep model loading, UI shell, accessibility, storage and transport in
  ordinary browser code. Natlang drives state and may generate the view plan.

The task board runs with a browser-local GGUF model selected from the catalog.
Its `?fixture` mode uses scripted model turns to verify wiring without model
weights. CPU-side tests execute the actual `.nl` source, event queue and DOM
projection. A Chromium end-to-end smoke is available through
`node scripts/browser-pilot.mjs --application`; it has not been completed on
the current workstation after an AMD display-driver failure during testing.

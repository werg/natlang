# Natlang-driven browser applications

A browser application is ordinary TypeScript whose reducer and view call
natural-language functions where judgment is needed. `EventLoop` gives one state
owner an explicit event order; `BrowserDomRenderer` turns checked view trees into
controls that emit the next events. Neither is a language primitive.

```ts
import * as natlang from '@natlang/browser';

const board = natlang.compileVirtualProject({ files: { 'board.ts': source } }, natlang).require('board.ts');
const runtime = natlang.createNatlangRuntime({ model: model.turn });
const loop = new natlang.EventLoop({
  initialState: board.initialBoard(), reduce: board.reduce, view: board.view,
  step: (fn, context) => runtime.run(fn, { signal: context.signal }),
  onCommit: commit => journal.persist(commit),
  onTransition: transition => renderer.render(transition.view),
  onFailure: failure => showFailure(failure.stage, failure.error),
});
const renderer = new natlang.BrowserDomRenderer(root, event => loop.dispatch(event));
await loop.start();
```

## Ordering and durability

- Events are applied one at a time in arrival order. Events that arrive during a step wait; they never rewrite a running call.
- An event ID is applied at most once per loop; restore `seenEventIds`, the state, and `initialRevision` from your journal for durable deduplication.
- `onCommit` is awaited before the new state is published or viewed. A failed commit leaves the previous state.
- A failed view leaves the committed state; `refresh()` recomputes the view without replaying the event.
- `cancel()` aborts the active step; `close()` stops the loop.
- Failed reductions can still have effects (service calls); preserving the previous state is not a rollback.

## UI choices

1. **Natlang reducer, exact view.** High-rate controls and exact layout; natlang interprets the events.
2. **Natlang view plan, exact renderer.** Natlang chooses grouping, wording, and visible actions; exact code checks IDs and coverage and builds the tree. The [task board](examples/browser-board/) works this way.
3. **Generated interfaces.** Natlang writes a view or module whose controls emit events to real, versioned handlers (the Studio research lab's generated modules run in an isolated iframe).

`BrowserDomRenderer` accepts `UiNode` trees (`tag`, `text`, `value`, `label`, `action`, `children`), renders text as text nodes, rejects executable tags, and emits typed `{ kind, value }` events. Keep keystrokes and animation exact; let natlang decide what submitted events mean.

## Examples

- `examples/browser-board/` — a TypeScript module with inline `nl` calls, compiled in the page (`?fixture` for scripted wiring).
- `examples/browser-local/` — model loading, diagnostics, and a three-task pilot.
- `playground/` — edit, run, and inspect projects with live interfaces and time travel.
- `studio/` — twenty-two applications and the research lab on one runtime, with a durable operation journal.

Run `npm run test:browser` for the real-Chromium smoke of the board and runtime.

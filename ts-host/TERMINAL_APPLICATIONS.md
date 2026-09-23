# natlang terminal applications

Terminal applications are TypeScript entry modules whose reducers call natural-
language functions. The TypeScript host supplies small utilities for the
lifecycle around them; none of them is a language feature.

- `EventLoop` applies events serially, suppresses duplicate IDs, awaits the commit before the view, retries a failed view with `refresh()`, and cancels the active step.
- `EventQueue` merges readline input, job completions, file watches, sockets, or other push sources; events wait while a step runs.
- `TerminalSessionStore` atomically checkpoints portable state, revision, and committed event IDs, and appends a reduction journal. It is a single-writer local store, not a lock or effect transaction.
- `renderTerminalView` renders checked headings, text, status, lists, tables, and code, stripping control characters from view content.
- `runTerminalShell` turns lines into typed events; `/help`, `/refresh`, `/cancel`, `/interrupt`, and `/quit` control the lifecycle, and applications register their own `/commands`.
- `openAICompatibleModelTurn` is the model transport; endpoint-specific aliases and request fields are configuration.

## A minimal application

```ts
import { join } from 'node:path';
import { EventLoop, TerminalSessionStore, runTerminalShell, type TargetContext, type TerminalView } from '@natlang/node';
import { answer, type State, type Request } from './index.js';

export async function main(context: TargetContext): Promise<number> {
  const store = new TerminalSessionStore<State, Request>(join(context.stateDirectory, 'session.json'));
  const saved = store.load({ questions: [], answers: [] });
  const loop: EventLoop<State, TerminalView, Request> = new EventLoop({
    initialState: saved.state, initialRevision: saved.revision, seenEventIds: saved.seen_event_ids,
    reduce: (state, event) => answer(state, event.value),
    view, step: fn => context.runtime.run(fn),
    onCommit: commit => store.commit(commit, loop.seenEventIds) });
  try {
    await runTerminalShell(loop, { input: context.io.input as never, output: context.io.output as never,
      event: (value, id) => ({ id, kind: 'request', value }),
      commands: { sources: { description: 'list loaded sources', run: () => listSources() } } });
  } finally { await loop.close(); }
  return 0;
}
```

Package it with a `natlang.json` target whose `entry` is this module and run it
with `natlang run DIRECTORY`; see [native packages](../NATIVE_PACKAGES.md).

## Events and jobs

Input and completion producers may run concurrently; reductions do not. A job
launch should return an ID promptly and publish its actual completion as a later
event. Cancellation records a request and observes the eventual result; it does
not assert rollback. After a restart, portable state and event IDs are restored;
native processes are not. The semantic terminal enqueues a `recover` event when
its checkpoint says a job was running and records an unknown outcome before
accepting more work.

`CommandRecipeLibrary` (in `applications/terminal/`) runs authored argument
vectors with `shell: false`, a checked working directory, bounded output, a
timeout, and abort propagation. It is trusted local execution, not a sandbox.

## Included applications

| Package | Directory | What natlang does |
|---|---|---|
| `@natlang/semantic-terminal` | `applications/terminal/` | Chooses an exact recipe for a request and explains actual outcomes |
| `@natlang/log-console` | `applications/logs/` | Judges each log event from exact evidence: ignore, investigate, or escalate |
| `@natlang/evidence-console` | `applications/evidence/` | Plans searches, picks passages, composes a cited answer; citations are verified |
| `@natlang/notebook-console` | `applications/notebook/` | Picks the goal cell, walks its dependencies with `iterateOn`, explains results |

```bash
natlang run applications/terminal
natlang run applications/evidence -- --documents evidence.json
natlang run applications/notebook -- --notebook notebook.json
cat logs.jsonl | natlang run applications/logs
```

Each starts with guided content; `/help` lists its commands. `evidence.json` is
an array of `{id,text}` documents, `notebook.json` holds `{cells,tables}`, and
each log line is one `LogEvent`.

## Verification

`test/terminal-application.test.mjs` covers the shell, session store, event
consumption, rendering, the model transport, and command recipes; each
application has its own test under `test/`. Fixture drivers establish wiring;
live-model quality and recovery against real external systems are separate
evaluation gates.

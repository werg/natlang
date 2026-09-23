# Terminal applications and packages

## A console application

```ts
import { EventLoop, TerminalSessionStore, runTerminalShell, type TargetContext, type TerminalView } from '@natlang/node';

export default async function main(context: TargetContext): Promise<number> {
  const store = new TerminalSessionStore<State, Request>(join(context.stateDirectory, 'session.json'));
  const checkpoint = store.load(initialState());
  const loop: EventLoop<State, TerminalView, Request> = new EventLoop({
    initialState: checkpoint.state, initialRevision: checkpoint.revision, seenEventIds: checkpoint.seen_event_ids,
    reduce: (state, event) => answer(state, event),            // ordinary code that calls natlang
    view, step: fn => context.runtime.run(fn),
    onCommit: commit => store.commit(commit, loop.seenEventIds) });
  try {
    await runTerminalShell(loop, { input: context.io.input, output: context.io.output,
      event: (value, id) => ({ id, kind: 'request', value }), commands: { /* /slash commands */ } });
  } finally { await loop.close(); }
  return 0;
}
```

- `TerminalSessionStore` is a single-writer checkpoint and event journal; restore state, revision, and `seen_event_ids` together. Native processes and handles are not restored: reduce a recovery event to an honest unknown outcome before retrying.
- `EventQueue` merges readline input with job completions, watchers, or sockets; events are reduced in queue order while each step runs.
- `TerminalView` blocks (text, status, list, table, code) render through `renderTerminalView`, which strips control characters. A failed event is reported at the prompt; the committed state remains.
- Return a job ID promptly for long work and publish its actual outcome as an event.

## Packages and `natlang run`

A `natlang.json` (`natlang.package/v2`) names the package, its included files, and targets: `{ entry, export?, description, authority, commands }`. The entry module's function (default `main`) receives a `TargetContext`: `args`, `io`, `workspace`, `stateDirectory`, `traceDirectory`, `model`, a configured `runtime`, and package identity.

```sh
natlang run applications/evidence -- --documents notes/   # build and run a package in place
natlang package pack natlang.json --root . --out app.nlpkg
natlang package install app.nlpkg && natlang run @scope/app
natlang setup && natlang doctor                             # managed local model runtime
```

The launcher builds the package against its own runtime, so the application and launcher share one runtime instance. `natlang call FILE.nl --inputs FILE` calls one named function; `natlang ask` answers an instruction over the working directory and the nearest `natlang.d/`.

Anchors: `applications/{evidence,logs,notebook,terminal}/`, `ts-host/src/terminal/`, `ts-host/src/package/`, `ts-host/src/cli/main.ts`, `NATIVE_PACKAGES.md`.

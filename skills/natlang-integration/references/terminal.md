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
- For long work, call `context.after(work)` in the reducer: it runs once the step is committed, and the event it returns is applied next, so the prompt stays responsive.

## Packages and `natlang run`

A `natlang.json` (`natlang.package/v2`) names the package, its included files, and targets: `{ entry, export?, description, authority, commands, arguments? }`. The entry module's function (default `main`) receives a `TargetContext`: `args`, `arguments`, `io`, `workspace`, `stateDirectory`, `traceDirectory`, `model`, a configured `runtime`, and package identity.

Declare a target's command-line `arguments` in order, each `{ name, type, description?, required?, default?, path? }` with a natlang `type` (the `.nl` signature grammar; `types.ts` aliases are in scope). `natlang run` checks the words after `--` against them before building and passes the values by name in `context.arguments`, with defaults applied and absent optional ones left out: read `context.arguments.tickets as Ticket[]` instead of parsing `context.args`. Words fill the arguments in order or name one (`--rubric "…"`, `--count=3`); a text type is written as is, any other type as JSON; an empty word skips an optional argument; `natlang inspect` prints the target's signature and usage. `path: "file" | "folder"` (type `string`) must exist in the workspace and arrives absolute. The dashboard builds the target's run form from the same declaration. Errors and their fixes:
- `count: expected number, got "x"`: write the value of the declared type; JSON for non-text types (`3`, `'["a","b"]'`).
- `tickets/0/priority: expected 1 | 2 | 3, got 5`: the path names the field inside the JSON value to correct.
- `json is not a natlang type; use unknown…`: declare `unknown`, or better the precise type the program reads.
- `an argument with a default is optional; remove required`: keep one of the two.
- `--colour is not an argument of this target; its arguments are …`: use a declared name, or `--name=--value` for a value that starts with `--`. A subdirectory with its own `natlang.json` inside another package is a nested package: the enclosing build skips its `.nl` files. A workspace root without `natlang.json` builds its member packages.

```sh
natlang run applications/evidence -- --documents notes/   # build and run a package in place
natlang package pack natlang.json --root . --out app.nlpkg
natlang package install app.nlpkg && natlang run @scope/app
natlang setup && natlang doctor                             # managed local model runtime
```

The launcher builds the package against its own runtime, so the application and launcher share one runtime instance. `natlang call FILE.nl --inputs FILE` calls one named function; `natlang ask` answers an instruction over the working directory and the nearest `natlang.d/`.

Anchors: `applications/{evidence,logs,notebook,terminal}/`, `ts-host/src/terminal/`, `ts-host/src/package/`, `ts-host/src/cli/main.ts`, `NATIVE_PACKAGES.md`.

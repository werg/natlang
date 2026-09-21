# Native natlang terminal applications

The TypeScript host includes a shared application layer for CLIs, terminal
dashboards, log consumers and long-running native jobs. A terminal application
has two ordinary natlang roots:

```text
reduce(state: State, event: Event) -> State
view(state: State) -> TerminalView
```

The reducer owns interpretation, planning, operation selection, inspection and
recovery decisions. Crisp helpers expose exact native operations and turn their
observations into typed values. The framework orders events, derives seeds,
commits completed reductions, retries presentation separately, suppresses event
IDs already committed and manages cancellation. It does not invent an action
plan or interpret process output.

## Components

Import the public API from `@natlang/typescript-host/terminal` or the package
root:

- `TerminalNatlangApplication` runs a serial reducer/view lifecycle over any
  `NativeNatlangHost`-compatible runner.
- `TerminalEventQueue` merges readline input, process completions, file watches,
  sockets or other push sources without interrupting a running reduction. The
  next event waits in the queue.
- `TerminalSessionStore` atomically checkpoints portable state, revision and
  committed event IDs, and appends a reduction journal. It is a single-writer
  local store, not a distributed lock or effect transaction.
- `renderTerminalView` renders checked headings, text, status, lists, tables and
  code. View-provided control characters are stripped. Natlang chooses the
  content; the renderer owns terminal mechanics.
- `runTerminalShell` turns lines into typed events while `/refresh`, `/cancel`,
  `/interrupt` and `/quit` control the application lifecycle. Applications can map `/cancel`
  to a semantic cancellation event for a native job.
- `openAICompatibleModelTurn` is a separate model transport. Endpoint-specific
  tool aliases and request fields are configuration, not source-language rules.

## Minimal application

```ts
import { NatlangHost, TerminalNatlangApplication, TerminalSessionStore,
  runTerminalShell } from '@natlang/typescript-host';

const native = { jobs, drainEvents: () => jobs.drainEvents() };
const host = new NatlangHost({ host: native, mode: 'retained' });
const store = new TerminalSessionStore<State>('session.json');
const saved = store.load(initialState);
let app: TerminalNatlangApplication<State, TerminalView, Event>;

app = new TerminalNatlangApplication({
  runner: host,
  source: { reducer: 'program/reduce.nl', view: 'program/view.ts' },
  initialState: saved.state,
  initialRevision: saved.revision,
  seenEventIds: saved.seen_event_ids,
  modelTurn,
  seedRoot: 17,
  traceDirectory: '.natlang/traces',
  onCommit: commit => store.commit(commit, app.seenEventIds),
});

await runTerminalShell(app, {
  event: (value, id) => ({ id, kind: 'request', value }),
});
host.close();
```

`onCommit` finishes before the framework publishes new state or computes its
view. A failed view can be retried with `refresh()` without rerunning the
reducer. A failed commit leaves the in-memory application at its previous
revision. Native effects may already have happened; applications that retry
effects need stable operation IDs and reconciliation in their host library.

## Event and job semantics

Input and completion producers may run concurrently. Reducer executions do not:
the application consumes events in queue order. This preserves the simple
lambda execution model while supporting responsive terminals. A native job
launch should return quickly with an ID. Its actual completion becomes another
typed event. Cancellation records a request and observes the eventual result;
it does not assert rollback.

After process restart, portable state and committed event IDs can be restored.
Native processes cannot. The semantic terminal sends a `recover` event when a
checkpoint says a job was running, and its natlang program records an unknown
outcome before accepting more work. Other applications should define their own
reconciliation event rather than silently rerunning work.

The default command recipe library executes authored argument vectors with
`shell:false`, a checked working directory, bounded captured output, a timeout
and abort propagation. It is trusted local execution, not a sandbox. Add Bash
as an explicit host recipe only when shell parsing and expansion are part of
the requested operation.

## Included applications

- `applications/semantic_terminal_cli.mjs` is a persistent interactive
  terminal. Natlang selects recipes, launches jobs, explains actual outcomes and
  handles recovery. The supplied workspace recipes inspect Git/files, run the
  Python suite and build the TypeScript host.
- `applications/log_console.mjs` consumes an async stream of typed log events.
  Natlang searches exact retained evidence and decides whether to ignore,
  investigate or escalate; the framework persists and renders each reduction.
- `applications/evidence_console.mjs` is an interactive cited research CLI.
  Each question runs the existing natlang search/select/read/compose/verify
  program and presents claims, literal citations and unresolved gaps.
- `applications/notebook_console.mjs` lets natlang select a goal cell from exact
  metadata, traverse the declared dependency graph, run SQL/TypeScript cells and
  explain bounded results in an interactive session.

The build and media workbenches are available as semantic-terminal recipes and
remain separate native libraries. This keeps their binary/process authority out
of the core framework while allowing natlang to compose them.

## Running the semantic terminal

Build the TypeScript host, then point the CLI at an already-running compatible
model endpoint:

```bash
npm --prefix ts-host run build
node applications/semantic_terminal_cli.mjs \
  --server http://127.0.0.1:8081 --model MODEL_ID \
  --root . --session .natlang/terminal-session.json \
  --traces .natlang/traces
```

The equivalent environment variables are `NATLANG_SERVER`, `NATLANG_MODEL` and
optional `NATLANG_API_KEY`. The CLI does not start or reconfigure a model
server. There are no framework-imposed trajectory or token limits; deployment
options can supply explicit budgets.

The other included CLIs use the same model flags:

```bash
node applications/evidence_console.mjs --documents evidence.json \
  --server http://127.0.0.1:8081 --model MODEL_ID
node applications/notebook_console.mjs --notebook notebook.json \
  --server http://127.0.0.1:8081 --model MODEL_ID
cat logs.jsonl | node applications/log_console.mjs \
  --server http://127.0.0.1:8081 --model MODEL_ID
```

`evidence.json` is an array of `{id,text}` documents. `notebook.json` contains
`{cells,tables}` using the `NotebookWorkspace` contract. Each log line is one
typed `LogEvent`. Add `--exchanges PATH` to retain the exact model transport
exchanges separately from reduction traces.

## Verification

`test/terminal-application.test.mjs` covers ordered reductions, durable event
IDs, rendering, model transport adaptation and an actual semantic-terminal job
completion. Existing semantic terminal, log investigator and evidence atlas
tests continue to exercise their native libraries and natlang algorithms.
Fixture drivers establish wiring. Live model quality, arbitrary command safety
and recovery against real external systems remain separate evaluation gates.

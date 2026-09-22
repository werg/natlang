# Native CLI and terminal applications

Use `TerminalNatlangApplication` from the TypeScript host for a native CLI with
the same reducer shape as a browser application:

```text
reduce(state: State, event: Event) -> State
view(state: State) -> TerminalView
```

Natlang should interpret requests, select and compose operations, inspect
results, explain failures and decide recovery. The framework serializes reducer
runs, commits state before presentation, derives event seeds, suppresses
committed event IDs and retries the view independently. Crisp helpers own exact
process/file/database calls and typed result checks.

For a terminal application that reasons over its working tree, declare a
`files: Dict<ProjectFile>` reducer argument and configure
`reducerInputs: () => ({ files: new NodeFileTree(root) })`. The reducer can
browse `args/files/...` lazily and pass the same dictionary to semantic helpers.
The factory creates a fresh dictionary for each reduction, while observations
within one run remain stable. Keep it out of a crisp view's inputs.

Use `TerminalEventQueue` to combine readline input, job completions, watchers or
sockets. Producers may be concurrent, but events wait while one lambda runs and
are reduced in queue order. This gives responsive jobs without a coroutine or
interrupt primitive in the language. Completion, cancellation and restart are
explicit typed events.

Use `TerminalSessionStore` for a local single-writer portable checkpoint and
event journal. Restore state, revision and `seen_event_ids` together. Native
processes and database handles are not restored. If a saved state names a
running operation, reconcile it through the host when possible or reduce a
recovery event to an honest unknown outcome before retrying.

Use `TerminalView` blocks for text, status, lists, tables and code, rendered by
`renderTerminalView`. The renderer owns ANSI/layout and strips supplied control
characters. Natlang can choose grouping and content in its view program. A
crisp view is appropriate when it only projects semantic state.

`runTerminalShell` maps input lines to application-defined events. Slash
commands control redraw, inference cancellation, semantic job cancellation and
exit. For jobs, return an ID promptly and publish their actual outcomes through
an event source. Shutdown must abort or transfer ownership of native work.
One failed event is reported at the prompt without terminating the shell; the
last committed state remains available for a corrected request or `/refresh`.

`openAICompatibleModelTurn` keeps endpoint/model aliases and raw exchanges in a
transport adapter. Configure it for the actual endpoint; do not copy those
quirks into `.nl` source. It deliberately does not start a server or impose
trajectory limits.

In a checkout, consult `ts-host/TERMINAL_APPLICATIONS.md`,
`ts-host/src/terminal/`, `applications/semantic_terminal_cli.mjs`,
`applications/log_console.mjs`, `applications/evidence_console.mjs`, and
`applications/notebook_console.mjs`.

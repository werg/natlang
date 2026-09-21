# Natlang terminal application framework

Status: first shared implementation complete, 2026-09-21.

## Design

Terminal applications use the same event-reducer idea as browser applications:
a typed event enters a queued `reduce(state,event)` run, the completed state is
durably committed, then `view(state)` produces presentation data. Incoming
events may be generated concurrently, while natlang reductions remain serial.
This supports process completions, logs and user input without adding interrupts
or coroutines to the language.

The framework is intentionally mechanical. Natlang owns semantic routing,
multi-step algorithms, interpretation of observations, explanations and
recovery choices. Host libraries own process/file/database objects, exact
validation, bounded reads, operation IDs and actual effects. Crisp view helpers
may project already-decided state. The terminal renderer owns ANSI and table
layout. Model transport remains independently replaceable.

## Implemented shared contracts

1. A queued application lifecycle with derived per-event seeds, local validation
   repair, cancellation, duplicate suppression and presentation retry.
2. A push async event queue for multiple producers. Queue order is observable;
   events never mutate an active lambda.
3. Atomic portable checkpoints and an append-only local event journal.
4. A typed terminal view value and control-character-safe renderer.
5. An interactive line shell with application-defined event construction.
6. An OpenAI-compatible adapter with explicit tool-name aliases and raw exchange
   hooks.
7. Structured command recipes with no shell, checked working directories,
   bounded output, timeout and abort propagation.

## Application integration

The semantic terminal now has reducer/view entrypoints and a runnable persistent
CLI. Its recipe jobs publish completion events, cancellation reaches native
processes, shutdown aborts owned jobs, and restart converts a lost running job
into an explicit unknown outcome.

The log investigator now has reducer/view entrypoints and a streaming console.
The evidence atlas and notebook now have interactive consoles that reuse their
complete semantic retrieval and dependency-execution programs. Build and media remain composable host libraries;
their work should enter the terminal as recipes rather than duplicated process
frameworks.

## Next empirical gates

- Run teacher and student scenarios covering ambiguous recipe selection,
  multi-operation recovery, late completion, user cancellation and restart.
- Add a persistent native operation ledger for recipes whose external systems
  support lookup/idempotency. The state checkpoint alone cannot settle unknown
  effects.
- Add bounded incremental output events when real scenarios show value beyond
  completion summaries. Coalesce high-volume output before it reaches natlang.
- Generalize the browser and terminal lifecycle onto one internal reducer core
  after parity tests demonstrate identical commit, duplicate and view-failure
  behavior. Their source loaders and presentation remain target-specific.
- Add richer terminal renderers only from demonstrated application needs: forms,
  selectable actions or full-screen diff panes should remain libraries over the
  typed view/event contract.
- Package application manifests when there are enough independently installed
  terminal apps to justify discovery. Do not add dependency resolution to the
  language runtime for this.

The strongest remaining CLI candidates are an IDE/code review console and
package-resolution console. Each should reuse this lifecycle and
its existing natlang algorithm. Their next work is application state/event
design and native recovery, not another CLI harness.

# P04 — Semantic terminal

Status: shared interactive terminal application implemented in
[`codebases/semantic_terminal`](../../codebases/semantic_terminal/README.md)
and [`applications/semantic_terminal_cli.mjs`](../../applications/semantic_terminal_cli.mjs).
It uses the shared queued reducer/view lifecycle, durable sessions, concurrent
job completion events, structured rendering and model transport documented in
[`TERMINAL_APPLICATIONS.md`](../../ts-host/TERMINAL_APPLICATIONS.md). Real build
and media recipes pass the Fold integration test; workspace Git/file/test/build
recipes pass through the CLI library. Live-model behavior and effect
reconciliation after restart remain open. [Shared capabilities](README.md).

## Natlang prerequisites

C0 supplies lexical recipes and semantic control flow. C1 supplies a retained host environment and explicit engines; C5 delivers user/process events to a Fold. C4 records operational evidence. C3 is only needed when the terminal later runs newly loaded natlang programmes. Neither a global shell tool nor in-episode interrupts are prerequisites.

## Programme and typed boundary

`step.nl(acc: Session, item: TerminalEvent) -> Session` delegates to `interpret.nl`, `choose_recipe.nl`, `execute_recipe.nl`, `explain.nl` and `recover.nl`. Link P01/P03 libraries into the appropriate companion functions rather than exposing every recipe in one menu. Begin with fixed imports; installed recipes become available in a subsequent invocation/revision.

`Session` contains current directory identity, environment summary, active request/revision, native-job IDs and outstanding results. Events distinguish user requests, process output summaries, completion and cancellation. Full logs and native subprocess objects remain outside the tree. Do not put a live shell object into serialised session state.

## Crisp environment

A session-scoped JS environment supplies filesystem/process/Git helpers. The `ts` engine can construct exact argument arrays and operate on native jobs directly. Add `bash` only for recipes that actually need shell semantics; required engine selection must not silently execute shell text as another language.

The host explicitly chooses shared or isolated execution. Shared access is allowed and carries that environment's authority; it must not be described as the QuickJS sandbox. Shell state persistence is deliberate: process working directory/environment and terminal session summaries must agree. A child command changing its own directory does not automatically change the enclosing session.

## Reduction and stream shape

One Fold owns each session. A long task can launch a job in a short step and consume later completion events. A new user request is another event; it can cancel/supersede work through an explicit session transition. It does not mutate an already-running lambda's instructions. Associate commands and late outputs with request IDs.

Small blocking commands can remain ordinary calls. Stream progress is coalesced by the host where appropriate; command completion and user requests are never silently discarded. Expose bounded output ranges through eval search/read helpers.

## Delivery and checks

The implemented stream path admits request and completion events, starts
session-owned recipe jobs without blocking the model step, and checks actual
host completion identity before updating session status. A cancel event records
the request and waits for the real result. Requests during an active job are
declined explicitly until resubmitted. The first integration composes P03 and
P01 through typed natlang selection, uses real processes, and checks forged
completion rejection. This confirms the current Fold boundary suffices for
sequential jobs; no mid-episode interrupt primitive is required by this case.

1. Execute a fixed recipe in a fixture workspace with exact command/result checks.
2. Compose a build diagnosis with P03, then a media request with P01. Gate: intermediate data passes through typed function boundaries correctly.
3. Retained session and background event behavior are implemented. Stale output
   is correlated, cancellation observes the actual result, and a restart turns
   an in-memory job into an explicit unknown outcome.
4. Add optional Bash and programme loading separately only for scenarios that
   require them, with engine/authority preflight.

Test quoting/metacharacters, missing executables, wrong directory, partial pipelines, large output and preserved unrelated Git changes.

## Trace and teacher

Record selected recipe, engine, relevant environment revision, commands, output observations and actual effects where capturable. Shared state limits replay unless a reset fixture is available. Train delegation to the right library, interpretation of failures, and continuing the user's task across completion events. A remote process proxy can later implement the same library interface; it is not mandatory for the local terminal.

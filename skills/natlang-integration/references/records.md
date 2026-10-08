# Call records and compilations

Every natlang runtime records each call in the machine's call store and can serve a call from a compilation of
recorded behavior. Repository design: `plans/TRACE_SPECIALIZATION.md`; code: `ts-host/src/calls/`.

## What is recorded

One `natlang.calls/1` record per call: the definition and its revision key, the parent call and which of its evals
started this one, exact inputs, captures read and written, the output, every service call with its exact arguments and
result, folder changes, the eval programs the agent ran, model requests, tokens and time, and the full trace events.
Values are content-addressed blobs; a value above `maxValueBytes` (1 MiB) is kept as its hash, type and a preview.

- Store: `$NATLANG_CALL_STORE`, default `~/.local/share/natlang/calls` (SQLite in WAL mode plus `blobs/`). One per
  machine; concurrent processes write to it. `NATLANG_CALL_STORE=off`, or `createNatlangRuntime({ calls: false })`,
  records nothing. Tests that must not touch the machine store pass `calls: false` or their own `CallStore.open(dir)`.
- Bounds: the store evicts when it passes `maxStoreBytes` (50 GiB) or its filesystem has less than `minFreeBytes`
  (20 GiB) free: first the event streams of the oldest calls, then whole calls. Calls cited by a compilation, annotated,
  queued for an audit, or pinned (`natlang traces pin CALL`) are kept.
- Opt-out per program in `natlang.json`: `"recording": { "exclude": ["auth/*.nl", "login.password"] }` (a definition
  source glob, or `functionName.argument`). Excluded values are recorded by type only; `return` names the result.
  `createNatlangRuntime({ recording: { exclude } })` does the same for an embedding.
- Give `createNatlangRuntime` the program's directory as `programRoot`, so offline work can reload its definitions.
  `natlang run` does this for packages.

## Reading records

```sh
natlang traces status                 # store, size, settings, pending jobs
natlang traces hot                    # definition revisions by calls, tokens or time
natlang traces list --definition support --executor crisp-agent
natlang traces show CALL              # inputs, output, effects, evals, cases, children
natlang traces export --definition support > calls.jsonl
natlang traces annotate CALL feedback '"the customer wanted a partial refund"'
```

From code: `CallStore.open(machineStoreRoot()!)` gives `calls(filter)`, `call(id)`, `events(id)`, `hot()`, `children(id)`,
`annotate(id, kind, value)` and `value(ref)`. Annotations are independent evidence for judging compiled cases.

## Compilations

A compilation is a stored `cases.ts` for one definition revision: guarded crisp cases (`when(args)`, `run(args)`)
written by the specializer from recorded calls. The program's source is never changed. When a call starts, the runtime
loads the current compilation for the definition's revision and context interface; the first active case whose guard
admits the call's arguments serves it, with the function's own context items and services. Otherwise the agent runs.

- A case that throws, returns a value of the wrong type, or throws `Deopt` after doing something hands the call to the
  agent, whose opening then says what the case already did (service calls, files, finished calls). `Deopt` before any
  effect simply lets the agent run.
- New cases start in shadow: the agent serves, and the call is replayed against the case offline. A case is promoted
  after `promotionComparisons` comparisons with at most `acceptanceBound` of them worse; an active case is demoted when
  audits or hand-offs exceed that bound. `auditRate` of crisp-served calls are re-run through the agent offline.
- Settings: `natlang traces config specialization=off|shadow|on auditRate=0.05 ...`, `NATLANG_SPECIALIZATION`, a
  program's `"specialization"` in `natlang.json`, and `createNatlangRuntime({ specialization })`; the lowest wins.

```sh
natlang specialize --loop                 # compile hot definitions, run shadow replays and audits, repeat
natlang compilations list
natlang compilations show support         # cases, tiers, numbers, cases.ts, the verification report
natlang compilations why CALL             # which case served a call, or why none did
natlang compilations export support DIR   # the compilation and its evidence as files
natlang compilations disable CASE
```

Effects of a crisp case are as real as the agent's: design services so the same operation is safe either way, and
use `ONCE_EFFECTS` for effects that must not repeat.

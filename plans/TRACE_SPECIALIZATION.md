# Trace-guided specialization: a tracing JIT for natural-language functions

Status: implemented 2026-10-08 (design revised three times that day with the owner's decisions, section 9). Code:
`ts-host/src/calls/` (store, recorder, normalization, mining, replay, judge, offline jobs, specializer support),
`ts-host/src/runtime/kernel.ts` (recording and dispatch), `ts-host/src/cli/calls.ts` (`natlang traces`,
`natlang compilations`, `natlang specialize`), `applications/specializer/`, `examples/specialization/`. Tests:
`ts-host/test/call-records.test.mjs`, `ts-host/test/specialization.test.mjs`. User documentation:
`skills/natlang-integration/references/records.md`, SPEC.md "Call records and compilations".

Where the implementation differs from the text below:

- The reducer's folder holds the evidence beside `cases.ts` (`evidence/...`), so the model reads it with ordinary file
  tools; only `cases.ts` is kept. The program is read through `evidence/function.md` and the function's context.
- §4: `iterateOn` site statistics moved into the store. The teacher collector keeps `exactHostTraceCapture` (its data
  contract is unchanged) and the improver keeps its evaluation traces; both can read the store, neither was rewritten.
- Offline jobs (shadow replays, audits) run in the specializer loop, with definitions reloaded from the recorded
  `program_root`, or rebuilt from the record (instructions, signature, types) when the source is unavailable.
- Bounds: besides `maxStoreBytes`, the store keeps `minFreeBytes` (20 GiB) free on its filesystem.
- Not built: the browser IndexedDB sink (§3.4, explicitly later).
- Replay answers a service call whose arguments differ from the record only in how a scalar is written (`999` and
  `"999"`); the call is observed with its own arguments, so the comparison shows the difference and the judge decides.
- Calls left `running` by a process that ended become `interrupted` when a store opens.

Live validation (2026-10-08, Qwen3.6-35B pi-executor): `examples/specialization` `support` got one case
(`/^refund \d+$/i` → `orders.refund`), accepted 6/6 on its admitted calls. It was promoted to active after 11 shadow
comparisons (0 worse) and then served `refund 777` in 29 ms (agent: ~7 s). Other requests went to the agent. nldb
`translate` was declined as unstable (identical inputs, three different schemas). The DGX runs the loop as the ledger
unit `natlang-specializer` (`scripts/specializer-loop.sh --profile pi-executor`, log
`~/.local/state/natlang/specializer.log`). A reducer pass takes 20–100 minutes on this executor.

## 1. Idea

A tracing JIT watches an interpreter. It finds hot paths, compiles each one under guards, and returns to the
interpreter ("side exit") when a guard fails. Here the interpreter is the agent (a model running a `.nl` function),
and the compiled path is crisp TypeScript.

The specializer does **not** compile instructions to code ahead of time. It reads recorded executions of a running
program and looks for invocation patterns that are already crisp in practice:

- *"When the request mentions `refund`, the executor always runs the same three-line eval against `orders.lookup`
  and returns `{ kind: 'refund', ... }`."*
- *"For inputs whose `items` array is empty, the call always returns `[]` after one turn."*
- *"When `format` is `'csv'`, the model always writes an equivalent split/join program."*

For each such pattern it writes a **case**: a crisp **guard** (`when`, a predicate over the call's typed inputs) and a
crisp **body** (`run`, a TypeScript function with the definition's signature). The cases for one definition form a
**compilation**, stored centrally beside the traces and loaded by the runtime without changing the program (section 7). At call time the runtime checks the guards.
When one holds, the body runs instead of the agent. If no guard holds, the agent runs as before. If the body fails,
the failure goes back to the agent, which finishes the call (section 6.5). The `.nl` function remains the
specification and the fallback.

Specialization is part of the runtime: recording, dispatch, the background specializer run, tier promotion and
auditing all happen without an application asking for them (section 7). The compiler itself is a natlang program
(`applications/specializer`): a directory reducer that writes a compilation folder from the program and its
recorded evidence. The runtime runs it offline over the machine's record store. Natural-language
functions make the judgments (which executions show the same approach, which conditions separate them, whether to
compile at all, how to write the body). Crisp code handles counting, verification, hashing and storage.

The same record store also serves the improver, training-data harvesting, cost profiling and debugging. That is why
much of this document is about instrumentation (sections 3–4).

## 2. What exists today

| Piece | Where | Useful for | Gap |
| --- | --- | --- | --- |
| Per-call trace recorder (`natlang.trace`) | `native/trace.ts`, `NativeRuntime.trace` | every event of one invocation | in memory only; delivered once at the end of the call |
| Exec graph (`natlang.exec-graph/1`) | `native/graph.ts`, spec/NEURALESE_GRAPH.md | typed nodes and edges, block producers | Neuralese-oriented; no index across calls |
| `InvocationTrace` + `TraceSink` | `runtime/runtime.ts:31` | call id, parent, definition id, outcome, adaptation provenance, events | no program build or source hash, no cost rollup, no link to the parent's action |
| `fileTraceSink` | `runtime/node-files.ts:36`; CLI `--trace`, apps `state/traces` | one JSONL file per call | no query, aggregation or retention; filenames are the only index |
| `state` events | `NativeRuntime.stateSummary` | initial and final state | **bounded previews**, not exact values |
| `action` events | `NativeSession` (`native/runtime.ts:1141`) | the model's eval code, result text, diagnostics | arguments over 100k chars become previews; no link to the child calls an eval started |
| `effect` events | `recordingServices` (`runtime/kernel.ts:403`) | service method requested/completed | results are not stored in a form a replay can serve back |
| Exact host capture | `exactHostTraceCapture` option | exact inputs/outputs | opt-in, needs a source allowlist and arguments listed by name; built for teacher capture |
| Iteration statistics | `runtime/iterate.ts` `IterationStatisticsStore` | per-site step stats | separate store, iterateOn only |
| Adaptation artifacts | `adaptation/`, spec/adaptation-artifact.schema.json | versioned overlays, compatibility checks, activation, rollback | component kinds are instruction text and guidance only |
| Rewrite gate | spec/NEURALESE_REWRITES.md, `compiler/rewrites.ts` | a rule is enabled only after a measured no-harm comparison | combinator-level only |

The kernel already gives us a single choke point: every natlang call goes through `invokeDefinition` →
`runDefinition` (`runtime/kernel.ts`). The guard check goes there.

## 3. Instrumentation: the call record store

Goal: every natlang execution on a machine is recorded by default, within declared bounds, in one store. The store
can be queried by program, definition, revision, site, time, outcome and cost. It holds exact typed inputs and
outputs, the approach the agent took, the effects it caused and what it cost. Recording is runtime behavior, not a
flag each application must remember to set.

### 3.1 Record shape (`natlang.calls/1`)

One row per invocation. Large values and the full event stream are content-addressed blobs.

```
call_id, parent_call_id, parent_action_seq      -- which eval of the parent started this call (new; see 3.3)
task_id, program_id, build_hash                 -- build_hash: hash of the compiled program (new)
definition_id, definition_source, revision      -- revision = hash of the instructions and contract as run
site: { kind: named | inline | iterate-step | judge,
        template_hash, slots }                  -- inline: hash of the un-interpolated template + slot values (new)
executor: { kind: agent | crisp | crisp→agent,  -- crisp→agent: a body failed and the agent finished (6.5)
            model_id, revision, dialect, temperature, seed_policy, case_hash }
adaptation                                      -- the instruction overlay that ran, if any
inputs_ref, output_ref, captures_ref            -- exact portable snapshots (blob hashes); see 3.2
input_features                                  -- typed feature vector, computed at record time (see 5.2)
outcome, detail
effects: [{ service, method, args_ref, result_ref, order, by: agent | crisp }]
folder: { base_hash, changed_paths, result_hash } -- directory reducers and folder arguments
capture_writes: [{ name, before_ref, after_ref }]
approach: { evals: [action seq…], approach_hash } -- normalized eval programs (see 5.1)
cost: { model_requests, tokens_in, tokens_out, wall_ms, turns, evals }
audit: { of_call, verdict }                     -- set on offline audit runs (6.3)
events_ref                                      -- the full natlang.trace JSONL as a blob
```

### 3.2 Bounded recording, on by default

`state` previews are not enough for guard mining or replay. The exact-capture code path (`exactPortableSnapshot`)
becomes the default for call inputs, captures read and written, service arguments and results, and outputs.

Two bounds apply, both in machine configuration with defaults:

- **Per value** (default 1 MiB). A larger value is recorded as its hash, its type and a preview, and is marked
  incomplete. A call with an incomplete input or output is still profiled. It can serve as mining evidence, but it
  cannot be a verification reference for that field.
- **Per store** (default 50 GB on NVMe). When the store passes its bound, eviction runs in this order: event-stream
  blobs of unpinned calls, oldest first; then value blobs of unpinned calls; then rows older than the retention
  horizon. Calls cited as evidence by a case, an adaptation artifact or a dataset manifest are pinned and
  are never evicted.

Programs opt out per definition or per argument in `natlang.json`
(`"recording": { "exclude": ["auth/*.nl", "login.password"] }`). Excluded values are recorded by type and hash only.

Live handles and streams are recorded as typed references. A folder is recorded as its content hash before and after,
plus the changed paths and their contents, which come under the per-value bound.

### 3.3 Linking calls to the eval that made them

To describe an approach ("ran this eval, which called `lookup` and then `summarize`"), a child call must name the
`action` that started it. `NativeSession` already knows the current action sequence number. It passes the number into
the frame (`Frame.parentActionSeq`), and `runDefinition` records it. A call tree then reads as
`eval code → child calls → their outputs → next eval`. That shape is the trace a specialization compiles.

### 3.4 One store per machine

- Location: `$NATLANG_CALL_STORE`, default `~/.local/share/natlang/calls`. On DGX and Pop it is set to
  `/home/werg/data/natlang/calls` on NVMe, never the external HDD.
- Format: SQLite (`node:sqlite`, already used by `skills/graded.ts`) in WAL mode for rows, so concurrent applications
  and tasks can write. A sibling `blobs/` directory holds SHA-256-named files for values and event streams.
- The runtime installs the sink by default (`createNatlangRuntime`), for the CLI, applications, evaluation workers
  and tests. A runtime option or `NATLANG_CALL_STORE=off` disables it. `--trace DIR` stays as a JSONL export.
- Writes happen in the existing end-of-call hook (`task.record`), so recording adds no latency to model turns. A
  failed write is reported once per task and never fails the call.
- Stores are machine-local and are not synchronized between machines. Compilations are content-addressed folders
  and can be copied between stores when that becomes useful.
  Records that become training data move as corpus manifests, like any other data.
- Browser runtime: an IndexedDB sink with the same row shape (later; not needed for the specializer).

### 3.5 Read access: the `traces` service and CLI

The store is exposed as an ordinary typed host service, so natlang programs (the specializer, the improver) read it
through the same capability mechanism as any other service:

```ts
traces.hot({ program?, since?, by: 'calls' | 'tokens' | 'wall_ms' }): HotDefinition[]
traces.calls({ definition, revision?, site?, outcome?, executor?, limit?, after? }): CallSummary[]
traces.call(callId): CallRecord                // with exact inputs, output, effects, approach
traces.events(callId): TraceEvent[]
traces.annotate(callId, { kind: 'judge' | 'feedback' | 'correction', value, source }): void
```

CLI: `natlang traces hot|list|show CALL|export --jsonl`.

### 3.6 Correctness signals

`outcome === 'done'` only says the call returned a typed value. Judging a case that behaves differently from the
agent (6.1) needs independent signals, attached after the fact with `traces.annotate`. The runtime records them automatically where it can: a caller
that catches a `NatlangCallError` and retries, an evaluation-suite score (`evaluation/`), a progress judge, an audit
verdict. Application-level feedback (a user edits the answer) goes through the same call.

### 3.7 Replay harness

`replay(callRecord, implementation)` runs a crisp body on the recorded exact inputs, captures and folder snapshot.
Each recorded service result is served back in order: a call with matching service, method and arguments receives the
recorded result. Anything else is a *replay divergence* and is reported, never silently allowed. Folder writes go to a
private copy. The harness compares the output, the capture writes, the folder result and the **effect sequence**
(service, method, arguments, order) with the record. Effects are part of the behavior being verified. This lets us
verify effectful bodies offline with no live services and no model. The harness reuses the evaluation runner's
isolation (worker, folder snapshots).

## 4. Instrumentation as a standard tool

Beyond the specializer, the store replaces several ad-hoc paths: the teacher collector's capture, improver evidence
gathering, `iterate` statistics (a view over rows), and debugging an application (`natlang traces show`). The plan
migrates these consumers after the store lands. Exact host capture keeps its narrow teacher semantics as a filter over
the store; it does not need a separate recording path.

## 5. Mining patterns

### 5.1 Approaches: normalized eval programs

For one definition revision, take the sequence of `eval` actions in each successful call. Normalize each program
(crisp): format it, replace each literal equal to an input value or input sub-path with that path, and alpha-rename
locals. Then **anti-unify** programs across calls: programs that agree after abstraction share an `approach_hash`, and
the anti-unifier is a template with holes bound to input paths. Child natlang calls inside the template stay calls.
The template is a partial specialization: a crisp skeleton with residual natural-language calls (like a JIT that
inlines some callees and leaves others). Service calls inside the template stay service calls, so the body causes the
same effects the agent did.

Where crisp normalization does not collapse two programs that do the same thing, an `.nl` judge
(`sameApproach(a, b): boolean`, a decision readout) merges clusters. Its merges are verified by replay before they
count.

Calls whose outputs follow from the inputs without any eval (the model just answered) form an approach of their own.
Their body has to be synthesized (5.3), not lifted.

### 5.2 Guards: cluster the conditions, leave the rest to the agent

Each call has typed input features, computed by type: enum/literal values, string membership of tokens and phrases
(from a vocabulary mined per field), regex classes (number, date, path, URL), lengths, numeric ranges, field presence,
array sizes, folder file extensions and counts, and the state of read captures.

The specializer is primed, in its instructions and in the shape of its results, to **cluster conditions**. For each
approach cluster it looks for the input conditions that select it, and it writes one guard per condition cluster.
Calls that no condition explains are **left unclassified**. They stay with the agent, and leaving them is a normal
result, not a failure. A guard that covers 30% of calls with no counterexample is better than one covering 90% with a
few.

Crisp rule induction over the features (small decision lists) produces the first candidate guards and measures every
candidate. The objective is precision first: a guard must have no counterexample in the training split, and recall
only decides whether the guard is worth having. An `.nl` function (`proposeGuards`) sees the clusters with examples and
the crisp candidates. It proposes conditions the features missed, merges or splits clusters, and names what each
condition means. Crisp code then measures each proposal's precision and coverage exactly. This is an `iterateOn` loop:
propose, measure, refine.

### 5.3 Declining is a first-class result

The specializer is explicitly invited to **decline**, per definition or per cluster, with a reason:

- `no-clusters`: approaches do not repeat, or no condition separates them (crisp: rule induction finds no guard
  above the minimum coverage at full precision; this is checked before any model is asked).
- `semantic`: the approach depends on understanding the input. A keyword may correlate with the approach in the
  recorded data while the agent actually decides by meaning. The `.nl` judge is asked this directly for every
  proposed guard: "would a reader decide this by the condition, or by what the text means?"
- `unstable`: recorded outputs for equal inputs disagree, or the approach changed over time.
- `effects`: the effects cannot be reproduced crisply (they depend on judgment at each step).
- `not-worth-it`: too little volume or cost to matter.

A decline is recorded with its evidence and is not revisited until the definition revision changes or the call volume
for that definition grows by a declared factor. Declines are themselves useful: they mark the functions where
natural language is essential.

Bodies may still contain residual natural-language calls. A cluster whose skeleton is crisp but which needs one
semantic judgment in the middle is specialized with that judgment left as a call, never replaced by a keyword
heuristic.

### 5.4 Bodies: lift, synthesize, improve

1. **Lift**: instantiate the anti-unified template as a TypeScript function. Residual natlang and service calls stay
   calls.
2. **Synthesize**: for answer-only approaches, an `.nl` author writes TypeScript from input/output/effect examples plus
   the instructions. Replay verifies the result.
Improving is not a separate mode. The compiler's instructions say that its crisp code may do better than the
recorded agent: merge two equivalent approaches, drop a wasted eval or a redundant service read, make inconsistent
clusters consistent, or handle inputs the agent often failed on. Acceptance (6.1) treats every case the same way.

## 6. Acceptance, lifecycle and runtime integration

### 6.1 Acceptance

There is one rule for every case, whether it copies the agent or improves on it. Replay (3.7) runs the case on a
held-out split of recorded calls its guard admits. For each call:

- If output, capture writes, folder result and effect sequence equal the recorded agent behavior, the call passes.
  Equality follows the return type, and the reference is the agreeing majority of recorded behaviors for equal
  inputs.
- If anything differs, an independent judge compares the two behaviors against the definition's instructions and
  contract, and against annotations on that call (3.6): `better`, `equivalent` or `worse`. The judge is a
  natural-language decision in the runtime, not part of the compiler, and it does not see which side is the case.

A case is accepted when no more calls than the acceptance bound are `worse`. Inputs whose recorded agent behavior
disagrees with itself are judged the same way and reported, since the disagreement is a finding about the program.
The same judge decides shadow and audit comparisons (6.2, 6.3).

### 6.2 Tiers, automatic and integrated

```
observed → candidate (verified offline) → shadow → active → (demoted)
```

The runtime moves cases through the tiers by evidence. No application has to activate them. A machine
setting (`specialization: off | shadow | on`, default `on`) and a per-program override in `natlang.json` limit how far
they may go.

- Promotion is automatic for every case, including ones that change behavior.
- **Shadow**: on guarded calls the agent runs and its result is returned. Afterwards, offline, the body is replayed
  against that call's record (recorded effects served back, no live effects). Divergences are recorded. Shadow adds no
  latency to the live call.
- **Active**: the body's result is returned. Audits (6.3) continue. A divergence or regression rate above the
  acceptance bound demotes the case automatically, and the demotion is recorded with the evidence.
- A case is bound to the definition revision, the build, the service declarations it calls, and the
  executor identity whose traces it was learned from. A change to any of them puts it back in shadow (newest code
  always: no frozen old case keeps running against new code).

### 6.3 Offline audits through the agent

A declared fraction of active, guarded calls (default 5%) is queued for audit. The live call returns the crisp result
immediately. Later, when executor capacity is idle, the auditor runs the agent on the recorded inputs, captures and
folder snapshot. Services are served from the crisp call's recorded effects where the arguments match. Where the agent
asks for a different effect, that is a divergence, and the effect is not performed. The auditor then compares the two
behaviors with the rule of 6.1 and annotates the original call.

Audit runs are ordinary records with `audit` set. They keep producing agent trajectories for guarded inputs, so
training data for those patterns keeps growing. They are marked so dataset builders can weight them deliberately.

### 6.4 Runtime dispatch

In `runDefinition`, before the agent is built: look up the definition revision's current compilation in the machine store
(7.3). Evaluate the guards of shadow and active cases in order, and run the first active case
whose guard admits. A guard that throws counts
as not admitting and is recorded. Body calls are recorded in the same store with `executor.kind = 'crisp'`. Nested
natlang calls from a body go through the kernel as usual and can be specialized in turn.

Bodies run with the call's own capabilities: its services (through `recordingServices`, so effects are journaled), its
folder transaction, and its capture cells. Capture writes are buffered and committed only when the body returns.

### 6.5 Errors go back to the agent

A body never ends a call with an error. When a body throws (any error, or an explicit `Deopt` when it meets an input
its guard admitted but it cannot handle), the runtime hands the call to the agent:

1. Buffered capture writes are discarded. Folder writes are not discarded: the folder transaction stays open, and
   the agent sees and continues from the files the body wrote.
2. Service effects that already happened cannot be undone. They are listed for the agent.
3. The agent runs the definition normally, with one note in its opening: the fast path for this call stopped with
   `<error>` after doing `<effects, changed files>`; continue from there and do not repeat finished effects unless
   the task needs them. The note is shown only in this case. There are no new tools, names or system-prompt text.
   This follows the existing practice of telling a retried phase why its first attempt failed.
4. The record is `executor.kind = 'crisp→agent'`, with the body's error and journal. Each hand-off counts against the
   case's divergence rate; repeated hand-offs demote it, and their records go to the specializer as
   counterexamples for the guard.

A body that returns a value of the wrong type is handled the same way: the type check fails, and the agent takes
over.

## 7. Compilations: stored centrally, loaded transparently

The program's source is never changed. Its `.nl` functions stay the specification and the fallback, and its folder
gains no files. A **compilation** is the specializer's output for one definition revision. It is stored in the
machine's record store, beside the calls it was derived from, and the runtime loads it at call time without the
program knowing.

### 7.1 What a compilation is

A compilation is a small content-addressed folder in the store:

```
<store>/compilations/<definition revision>/<compilation id>/
  cases.ts        the crisp cases (below)
  meta.json       definition id and source, program and build seen, context interface hashes,
                  executor identity of the evidence, parent compilation, created, specializer revision
  report.md       the specializer's account: groups found, cases written, calls left to the agent, declines
```

`cases.ts` is ordinary TypeScript written by the compiler:

```ts
// Crisp cases for database/classify.nl. Inputs no case takes go to classify.nl.
export const cases = [
  { when: (request: string) => /^\s*(create|define)\s+table\b/i.test(request),
    run: async (request: string): Promise<Classification> => ({ kind: 'schema' }) },
  { when: (request: string) => /^\s*(select|with)\b/i.test(request),
    run: async (request: string): Promise<Classification> => ({ kind: 'question' }) },
];
```

- `when` takes the function's parameters and returns a boolean. It must not cause effects.
- `run` has the function's signature. It runs with the function's own context and services, and nothing more, so a
  case never has more authority than its function. Calling a natlang item from the context is how a case keeps a
  semantic judgment natural (5.3).
- A `throw` in `run` (or `Deopt`, for an input a guard admitted but the case cannot handle) hands the call to the
  agent (6.5).
- Each case has a content hash. Tier state and all statistics are keyed by it.

Nobody writes these files by hand and no program imports them, so there is no new authoring concept and nothing
changes for program authors or for executing models.

### 7.2 Stored in relation to the traces

The store's tables link compilations, cases and calls in both directions:

```
compilations(id, definition_revision, definition_id, folder_hash, parent_id, created, status)
cases(hash, compilation_id, position, tier, admitted, served, handed_off, audited, worse, demoted_at)
case_calls(case_hash, call_id, role)   -- role: group | training | held-out | counterexample | shadow | audit
declines(definition_revision, reason, why, evidence_calls, revisit_after)
calls.case_hash                        -- set on every call a case served or shadowed (3.1)
```

So from a case one can reach every call it was learned from, checked against, served, handed back or was audited on.
From a call one can reach the case that served it. Calls cited by a case are pinned against eviction (3.2).

### 7.3 Loading at runtime

When a call starts, `runDefinition` looks up the current compilation for the definition's revision in the machine
store (cached in memory, refreshed when the store changes). A compilation applies only when its recorded context
interface matches the call's: the callees a case may call have the same signatures and revisions. Otherwise it is
not used, and the specializer is asked to recompile. The cases module is loaded through the callable-folder module
loader, bound to the definition's context and services. No program configuration is needed. The machine setting
(`specialization: off | shadow | on`) and a per-program override decide whether compilations are used.

Because the key is the definition revision, a compilation serves every program and build that contains the same
definition and context interface.

### 7.4 Inspecting compilations

A standard view comes with the store, at three levels:

- **CLI**: `natlang compilations list [--program P]` (definitions with compilations, tiers, coverage, saved time,
  hand-offs, declines); `natlang compilations show DEFINITION` (the cases source, each case's guard in words from
  the report, its numbers, and links to example calls); `natlang compilations why CALL` (which case served or
  declined a call, and why); `natlang compilations calls CASE --role held-out|served|audit|…`;
  `natlang compilations history DEFINITION` (compilations over time, demotions, declines); and
  `natlang compilations export DEFINITION DIR` (the folder plus a rendered evidence folder, for reading or review).
- **`traces` service**: the same queries for natlang programs (`traces.compilation(definition)`,
  `traces.caseCalls(hash, role)`), so the specializer and the improver read compilations the way they read calls.
- **`natlang traces show CALL`** shows the case that served the call next to its inputs and output.

A developer can also disable a case or a compilation from the CLI (`natlang compilations disable CASE`). This is
recorded like a demotion.

### 7.5 The specializer: a directory reducer over the compilation folder

The specializer edits a compilation folder, not the program:

```
applications/specializer/
  specialize.nl            directory reducer over a compilation folder; reads the program and the evidence;
                           writes or edits cases.ts and report.md, or declines
  specialize/
    sameApproach.nl        decision: do two eval programs do the same thing
    semanticCheck.nl       decision: is a condition how the agent really decides, or does it need meaning
  index.ts                 crisp: hot list, evidence folder rendering (7.6), replay verification, store writes
```

```ts
specialize(compilation: Folder, program: Folder, evidence: Folder, definition: string): Promise<
  | { kind: 'specialized'; cases: number; unclassified: number }
  | { kind: 'declined'; reason: 'no-clusters' | 'semantic' | 'unstable' | 'effects' | 'not-worth-it'; why: string }>
```

The compilation folder is the writable one; it starts as a copy of the current compilation, or empty. The program
and the evidence are read-only. On success the crisp top level stores the committed folder as a new compilation whose
cases start in shadow, and records the case-call links. On a decline it records the decline.

Its instructions say, in plain words:
- Group the recorded calls by the conditions that select each approach.
- Write one case per group whose condition you can state crisply.
- Leave every call you cannot classify to the natural-language function.
- Improve on what the agent did where you can.
- Decline, with a reason, when there are no groups or when the choice depends on meaning. Writing no cases is a good
  result when that is the truth.

Each definition is one `iterateOn` loop over the compilation folder. The step writes or edits `cases.ts`. Crisp code
then replays it (3.7, 6.1) and writes the report into the evidence folder: counterexamples per case, judge verdicts,
calls the guard wrongly admitted. The loop ends when every remaining case is accepted, or with a decline. A case that
cannot be made acceptable is removed, and its calls count as unclassified.

The top level is ordinary TypeScript in the application: take `traces.hot()`, skip definitions with a standing
decline (5.3), and call `specialize` for each remaining one.

### 7.6 The evidence folder

The model does not have to query the store. After the crisp mining steps of section 5, crisp code renders the
evidence as files:

```
evidence/
  function.md                 instructions, signature, callees, call volume and cost
  approaches/a1/approach.ts   anti-unified template with holes named by input path
  approaches/a1/examples/     a few exact records: inputs.json, output.json, effects.json, evals.ts
  approaches/a1/stats.md      calls, outcomes, recorded-behavior agreement
  conditions.md               crisp rule-induction candidates per approach, with precision and coverage
  unclassified/               calls in no approach or matching no condition
  history.md                  earlier compilations, declines, hand-offs, demotions, audit divergences
  report.md                   written by each replay round (7.5)
```

Every example names its call id, so whatever the compiler cites can be traced back. The model then does what it is
good at with ordinary file tools: it reads the examples, compares approaches and writes TypeScript. It can still
query `traces` when it wants more examples. The same rendering is what `natlang compilations export` writes.

### 7.7 Runtime loop

A machine-level background service runs the specializer when the store has enough new calls for a hot definition. It
runs the shadow replays and audits when executor capacity is idle. On DGX the loop goes through the memory ledger like
every other model-loading job, and it yields to training. It uses the newest code and data each time it starts, and it
can be interrupted and resumed at any step: its progress is records in the store.

Per the port-granularity rule, each part gets an explicit decision (NL function / instruction / crisp helper) in a
short DECOMPOSITION.md before implementation.

## 8. Order of work

1. Call record store: row schema, blobs, bounds and pinning, the per-machine sink installed by default, exact capture,
   effect results, folder and capture-write records, `parentActionSeq`, inline template identity, cost rollup. Tests:
   round trip, dedupe, bounds and eviction, pins, parent-action links, opt-out, concurrent writers.
2. `traces` service and `natlang traces` CLI. Migrate `iterate` statistics and the improver's evidence reads.
3. Replay harness with recorded-effect serving, folder copies, effect-sequence comparison and divergence reports.
4. Compilations in the store (tables of 7.2), runtime loading (7.3), dispatch, the error hand-off to the agent (6.5), the
   comparison judge (6.1), shadow replay, the audit queue, automatic tiers, and `natlang compilations` (7.4).
5. Specializer application: crisp mining and evidence-folder rendering first (normalization, anti-unification,
   features, rule induction, the crisp `no-clusters` decline), then the `specialize` reducer, its replay loop and the
   `.nl` decisions.
6. Background loop and machine settings.
7. First targets: pick high-volume definitions from recorded application runs. Likely candidates are nldb's
   `classify` and `filter/meets`, pi's per-message policies and the compilers' pattern-heavy passes. Report
   coverage, declines with reasons, hand-offs, divergence and saved model time per target.
8. Skills and docs: `skills/natlang-integration` covers recording, opt-outs, `traces`, specialization settings and
   the hand-off note. `skills/natlang-authoring` covers the recording opt-out in `natlang.json`.
   SPEC.md states that the runtime may serve a call from a compilation and hands failures to the agent.

## 9. Decisions

Owner decisions, 2026-10-08:

1. Recording is bounded and on by default, in one store per machine (3.2, 3.4).
2. Cases may improve on the agent. There is no separate mode: the compiler is told it may improve, and one
   acceptance rule judges every case (6.1). Automatic promotion applies to all cases.
3. Side-effecting functions are in scope: effects are journaled, replayed and compared (3.7, 6.4, 6.5).
4. The specializer is primed to cluster conditions into guards, to leave unclassifiable calls to the agent, and to
   decline when there are no clusters or when semantic understanding is needed (5.2, 5.3, 7).
5. Specialization is integrated into the runtime, and body errors go back to agent execution (6.2, 6.4, 6.5).
6. A sample of guarded calls keeps running through the agent, offline so live latency does not change (6.3).

7. Compilations are stored centrally in the machine store, linked to the calls they derive from and serve, loaded
   transparently at runtime, and inspectable through a standard CLI and service. Program source is never changed.
   The specializer is a directory reducer over a compilation folder (section 7).

Still open:

- Default bounds: 1 MiB per value, 50 GB per store, 5% audit rate. These are starting points to adjust when
  measurements arrive.

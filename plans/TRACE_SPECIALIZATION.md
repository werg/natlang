# Trace-guided specialization: a tracing JIT for natural-language functions

Status: design, 2026-10-08, revised twice the same day with the owner's decisions (section 9). Nothing below is
implemented.

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
crisp **body** (`run`, a TypeScript function with the definition's signature). A function's cases live in a
`foo.cases.ts` file beside `foo.nl` (section 7). At call time the runtime checks the guards.
When one holds, the body runs instead of the agent. If no guard holds, the agent runs as before. If the body fails,
the failure goes back to the agent, which finishes the call (section 6.5). The `.nl` function remains the
specification and the fallback.

Specialization is part of the runtime: recording, dispatch, the background specializer run, tier promotion and
auditing all happen without an application asking for them (section 7). The compiler itself is a natlang program
(`applications/specializer`): a directory reducer from the program's context and its recorded evidence to a new
context revision that contains cases files. The runtime runs it offline over the machine's record store. Natural-language
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
- Stores are machine-local and are not synchronized between machines. Cases files are portable source.
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

In `runDefinition`, before the agent is built: look up the cases attached to the definition in the program's bound
context revision (7.4). Evaluate the guards of shadow and active cases in order, and run the first active case
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

## 7. Shape in natlang: a reducer from context and evidence to a new context

The spec already names this operation: "Self-improvement is a function from a context and evidence to a new
context revision; promotion binds a program to that revision" (SPEC.md, Directory reducers). Specialization is one
such function. It needs no new execution concept. Its parts are a cases file, a directory reducer, an evidence folder
and context promotion.

### 7.1 The cases file

A function's crisp cases are an ordinary callable-folder TypeScript module beside its `.nl` file:

```ts
// database/classify.cases.ts: crisp cases for classify.nl. Inputs no case takes go to classify.nl.
import type { Classification } from '../types';

export const cases = [
  { when: (request: string) => /^\s*(create|define)\s+table\b/i.test(request),
    run: async (request: string): Promise<Classification> => ({ kind: 'schema' }) },
  { when: (request: string) => /^\s*(select|with)\b/i.test(request),
    run: async (request: string): Promise<Classification> => ({ kind: 'question' }) },
];
```

- `when` takes the function's parameters and returns a boolean. It must not cause effects.
- `run` has the function's signature. It may call what the function itself may call (the items of its context and
  its services) and nothing more, so a case never has more authority than its function. Calling a natlang item is
  how a case keeps a semantic judgment natural (5.3).
- A `throw` in `run` (or `Deopt`, for an input a guard admitted but the case cannot handle) hands the call to the
  agent (6.5).
- The runtime keys tier state by each case's content hash, not by its position. The compiler writes a comment line
  with the case's evidence (`// 412 calls, 0 worse, a1`).

The loader attaches `foo.cases.ts` to `foo.nl`. It is not an item that callers of the folder can call, and the agent
executing `foo` does not see it. Authors can write cases files by hand. The specializer reads and edits existing
ones like any other source, and hand-written cases go through the same acceptance and tiers.

This is the one new authoring concept in the design. It is ordinary TypeScript with two fields, so small models can
write it, and executing models never see it. `skills/natlang-authoring` gains a short section on it.

### 7.2 The specializer: a directory reducer

```
applications/specializer/
  specialize.nl            directory reducer over the program context; reads evidence; for one definition,
                           writes or edits foo.cases.ts, or declines
  specialize/
    sameApproach.nl        decision: do two eval programs do the same thing
    semanticCheck.nl       decision: is a condition how the agent really decides, or does it need meaning
  index.ts                 crisp: hot list, evidence folder rendering (7.3), replay verification, decline records
```

The program context comes first, so it is a directory reducer. Evidence is a second, read-only folder:

```ts
specialize(program: Folder, evidence: Folder, definition: string): Promise<
  | { kind: 'specialized'; cases: number; unclassified: number }
  | { kind: 'declined'; reason: 'no-clusters' | 'semantic' | 'unstable' | 'effects' | 'not-worth-it'; why: string }>
```

Its instructions say, in plain words:
- Group the recorded calls by the conditions that select each approach.
- Write one case per group whose condition you can state crisply.
- Leave every call you cannot classify to the natural-language function.
- Improve on what the agent did where you can.
- Decline, with a reason, when there are no groups or when the choice depends on meaning. Writing no cases is a good
  result when that is the truth.

Each definition is one `iterateOn` loop over the reducer's folder. The step writes or edits the cases file. Crisp code
then replays it (3.7, 6.1) and writes the report into the evidence folder: counterexamples per case, judge verdicts,
calls the guard wrongly admitted. The loop ends when every remaining case is accepted, or with a decline. A case that
cannot be made acceptable is removed, and its calls count as unclassified.

The top level is ordinary TypeScript in the application: take `traces.hot()`, skip definitions with a standing
decline (5.3), and call `specialize` for each remaining one on the same staged folder.

### 7.3 The evidence folder

The model does not have to query the store. After the crisp mining steps of section 5, crisp code renders the
evidence as files:

```
evidence/database/classify/
  function.md                 instructions, signature, callees, call volume and cost
  approaches/a1/approach.ts   anti-unified template with holes named by input path
  approaches/a1/examples/     a few exact records: inputs.json, output.json, effects.json, evals.ts
  approaches/a1/stats.md      calls, outcomes, recorded-behavior agreement
  conditions.md               crisp rule-induction candidates per approach, with precision and coverage
  unclassified/               calls in no approach or matching no condition
  history.md                  earlier declines, hand-offs, demotions, audit divergences
  report.md                   written by each replay round (7.2)
```

The model then does what it is good at with ordinary file tools: it reads the examples, compares approaches and
writes TypeScript. It can still query `traces` when it wants more examples.

### 7.4 Promotion and export

The runtime's background loop (7.5) calls `specialize` on a staged copy of the program's current context and keeps the
staged tree. Compiled, that tree is a new context revision of the program. The runtime binds the program to it on this
machine, and the new cases start in shadow. Tiers are runtime state per case hash; the revision only says which cases
exist.

Applying the same reducer with `folder.apply` writes the cases files into the source tree, where they are reviewed and
committed like any other change. That is the export. It is also the pluggable-hot-paths form: crisp cases beside the
natural-language function, chosen per call by their guards. A machine whose program source already contains cases
files starts from them.

The adaptation machinery (`adaptation/compatibility.ts`, `optimization/promotion.ts`) is reused for revision
bookkeeping and rollback. There is no separate artifact format.

### 7.5 Runtime loop

A machine-level background service runs the specializer when the store has enough new calls for a hot definition. It
runs the shadow replays and audits when executor capacity is idle. On DGX the loop goes through the memory ledger like
every other model-loading job, and it yields to training. It uses the newest code and data each time it starts, and it
can be interrupted and resumed at any step: its progress is records in the store and staged revisions.

Per the port-granularity rule, each part gets an explicit decision (NL function / instruction / crisp helper) in a
short DECOMPOSITION.md before implementation.

## 8. Order of work

1. Call record store: row schema, blobs, bounds and pinning, the per-machine sink installed by default, exact capture,
   effect results, folder and capture-write records, `parentActionSeq`, inline template identity, cost rollup. Tests:
   round trip, dedupe, bounds and eviction, pins, parent-action links, opt-out, concurrent writers.
2. `traces` service and `natlang traces` CLI. Migrate `iterate` statistics and the improver's evidence reads.
3. Replay harness with recorded-effect serving, folder copies, effect-sequence comparison and divergence reports.
4. Cases files: loader attachment to `foo.nl`, runtime dispatch, the error hand-off to the agent (6.5), the
   comparison judge (6.1), shadow replay, the audit queue, automatic tiers, revision binding per machine.
5. Specializer application: crisp mining and evidence-folder rendering first (normalization, anti-unification,
   features, rule induction, the crisp `no-clusters` decline), then the `specialize` reducer, its replay loop and the
   `.nl` decisions.
6. Background loop and machine settings.
7. First targets: pick high-volume definitions from recorded application runs. Likely candidates are nldb's
   `classify` and `filter/meets`, pi's per-message policies and the compilers' pattern-heavy passes. Report
   coverage, declines with reasons, hand-offs, divergence and saved model time per target.
8. Skills and docs: `skills/natlang-integration` covers recording, opt-outs, `traces`, specialization settings and
   the hand-off note. `skills/natlang-authoring` covers cases files and the recording opt-out in `natlang.json`.
   SPEC.md gains cases files under Natural-language functions.

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

7. Shape: a directory reducer from the program context and an evidence folder to a new context revision with
   `foo.cases.ts` files (section 7).

Still open:

- Default bounds: 1 MiB per value, 50 GB per store, 5% audit rate. These are starting points to adjust when
  measurements arrive.
- The `foo.cases.ts` name and its `when`/`run` fields (7.1), the one new authoring concept.

# Trace-guided specialization: a tracing JIT for natural-language functions

Status: design proposal, 2026-10-08. Nothing below is implemented. Section 9 lists the decisions for the owner.

## 1. Idea

A tracing JIT watches an interpreter. It finds hot paths, compiles each one under guards, and returns to the
interpreter ("side exit", "deopt") when a guard fails. Here the interpreter is a model running a `.nl` function, and
the compiled path is crisp TypeScript.

The specializer does **not** compile instructions to code ahead of time. It reads recorded executions of a running
program and looks for invocation patterns that are already crisp in practice:

- *"When the request mentions `refund`, the executor always runs the same three-line eval against `orders.lookup`
  and returns `{ kind: 'refund', ... }`."*
- *"For inputs whose `items` array is empty, the call always returns `[]` after one turn."*
- *"When `format` is `'csv'`, the model always writes an equivalent split/join program."*

For each such pattern it emits a **specialization**: a crisp **guard** (a predicate over the call's typed inputs) and a
crisp **body** (a TypeScript function with the definition's signature). At call time the runtime checks the guards.
When one holds, the body runs instead of the model. If no guard holds, or the body declines, the natural-language
function runs as before. The `.nl` function remains the specification and the fallback.

The specializer is itself a natlang application (`applications/specializer`). It follows the program-improver
conventions: natural-language functions make the judgments (which executions show the same approach, what the guard
means, how to write the body), and crisp code handles counting, verification, hashing and storage.

The same record store also serves the improver, training-data harvesting, cost profiling and debugging. That is why
most of this document is about instrumentation (sections 3–4).

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

Goal: every natlang execution is recorded by default in one store that can be queried by program, definition,
revision, site, time, outcome and cost. The store must hold exact typed inputs and outputs, the approach the model
took, the effects it caused and what it cost. Recording becomes runtime behavior, not a flag each application must
remember to set.

### 3.1 Record shape (`natlang.calls/1`)

One row per invocation. Large values and the full event stream are content-addressed blobs.

```
call_id, parent_call_id, parent_action_seq      -- which eval of the parent started this call (new; see 3.3)
task_id, program_id, build_hash                 -- build_hash: hash of the compiled program (new)
definition_id, definition_source, revision      -- revision = hash of the instructions and contract as run
site: { kind: named | inline | iterate-step | judge,
        template_hash, slots }                  -- inline: hash of the un-interpolated template + slot values (new)
executor: { model_id, revision, dialect, temperature, seed_policy }
adaptation, specialization                      -- the overlay and the specialization (if any) that ran
inputs_ref, output_ref, captures_ref            -- exact portable snapshots (blob hashes); see 3.2
input_features                                  -- typed feature vector, computed at record time (see 5.2)
outcome, detail
effects: [{ service, method, args_ref, result_ref, order }]
approach: { evals: [action seq…], approach_hash } -- normalized eval programs (see 5.1)
cost: { model_requests, tokens_in, tokens_out, wall_ms, turns, evals }
events_ref                                      -- the full natlang.trace JSONL as a blob
```

### 3.2 Exact values by default, with a declared opt-out

`state` previews are not enough for guard mining or replay. The exact-capture code path
(`exactPortableSnapshot`) becomes the default for named and inline call inputs, captures read, and outputs. Programs
opt out per definition or per argument in `natlang.json`:

```json
"recording": { "values": "exact", "exclude": ["auth/*.nl", "login.password"] }
```

Values that are not portable (live handles, folders, streams) are recorded as typed references: a folder as its
transaction's content hash plus the list of changed paths, a stream as `$opaque`. Such calls are still profiled, but
they are not candidates for specialization in v1.

There is no byte cap and no sampling until storage becomes a measured problem (language-design-restraint). Blobs are
deduplicated by content hash, which keeps repeated inputs cheap.

### 3.3 Linking calls to the eval that made them

To describe an approach ("ran this eval, which called `lookup` and then `summarize`"), a child call must name the
`action` that started it. `NativeSession` already knows the current action sequence number. It passes the number into
the frame (`Frame.parentActionSeq`), and `runDefinition` records it. A call tree then reads as
`eval code → child calls → their outputs → next eval`. That shape is the trace a specialization compiles.

### 3.4 Storage and sinks

- `callStoreSink(directory)`: SQLite (`node:sqlite`, already used by `skills/graded.ts`) for rows; a sibling
  `blobs/` directory of SHA-256-named files for values and event streams. Stores live on NVMe
  (`/home/werg/data/...` on DGX), never on the external HDD.
- The CLI and application runner install it by default under the state directory, replacing the per-call JSONL
  directory. `--trace DIR` stays as an export format.
- Writes happen in the existing end-of-call hook (`task.record`), so recording adds no latency to model turns. A
  failed write is reported once per task and never fails the call.
- Browser runtime: an IndexedDB sink with the same row shape (later; not needed for the specializer).

### 3.5 Read access: the `traces` service and CLI

The store is exposed as an ordinary typed host service, so natlang programs (the specializer, the improver) read it
through the same capability mechanism as any other service:

```ts
traces.hot({ program, since?, by: 'calls' | 'tokens' | 'wall_ms' }): HotDefinition[]
traces.calls({ definition, revision?, site?, outcome?, specialization?, limit?, after? }): CallSummary[]
traces.call(callId): CallRecord                // with exact inputs, output, effects, approach
traces.events(callId): TraceEvent[]
traces.annotate(callId, { kind: 'judge' | 'feedback' | 'correction', value, source }): void
```

CLI: `natlang traces hot|list|show CALL|export --jsonl`.

### 3.6 Correctness signals

`outcome === 'done'` only says the call returned a typed value. For "faithful" acceptance (6.1) recorded outputs are
the reference. For "improving" acceptance we need independent signals, attached after the fact with
`traces.annotate`. The runtime records them automatically where it can: a caller that catches a `NatlangCallError`
and retries, an evaluation-suite score (`evaluation/`), a progress judge. Application-level feedback (a user edits
the answer) goes through the same call.

### 3.7 Replay harness

`replay(callRecord, implementation)` runs a crisp candidate on the recorded exact inputs. Each recorded service
result is served back in order: matching arguments receive the recorded result; anything else is a *replay
divergence* and is reported, never silently allowed. This lets us verify a body offline with no live services and no
model. The harness reuses the evaluation runner's isolation (worker, folder snapshots).

## 4. Instrumentation as a standard tool

Beyond the specializer, the store replaces several ad-hoc paths: the teacher collector's capture, improver evidence
gathering, `iterate` statistics (a view over rows), and debugging an application (`natlang traces show`). The plan
migrates these consumers after the store lands. Exact host capture keeps its narrow teacher semantics as a
filter over the store; it does not need a separate recording path.

## 5. Mining patterns

### 5.1 Approaches: normalized eval programs

For one definition revision, take the sequence of `eval` actions in each successful call. Normalize each program
(crisp): format it, replace each literal equal to an input value or input sub-path with that path, and alpha-rename
locals. Then **anti-unify** programs across calls: programs that agree after abstraction share an `approach_hash`, and
the anti-unifier is a template with holes bound to input paths. Child natlang calls inside the template stay calls.
The template is a partial specialization: a crisp skeleton with residual natural-language calls (like a JIT that
inlines some callees and leaves others).

Where crisp normalization does not collapse two programs that do the same thing, an `.nl` judge
(`sameApproach(a, b): boolean`, a decision readout) merges clusters. Its merges are verified by replay before they
count.

Calls whose outputs follow from the inputs without any eval (the model just answered) form an approach of their own.
Their body has to be synthesized (5.3), not lifted.

### 5.2 Guards: predicates that predict the approach

Each call has typed input features, computed by type: enum/literal values, string membership of tokens and phrases
(from a vocabulary mined per field), regex classes (number, date, path, URL), lengths, numeric ranges, field presence,
array sizes, and folder file extensions/counts. Guard search is rule induction over these features (crisp; small
decision lists) that predicts the approach cluster. The objective is **precision first**: a guard must have no
counterexample in the training split. Recall only decides whether the guard is worth having, because a call the guard
does not catch simply falls back to the model.

An `.nl` function proposes candidate guards from examples ("all of these mention a refund and carry an order id"),
including features the vocabulary missed. Crisp code then measures each proposal's precision and coverage exactly.
This is an `iterateOn` loop: propose, measure, refine.

### 5.3 Bodies: lift, synthesize, improve

1. **Lift**: instantiate the anti-unified template as a TypeScript function. Residual natlang calls stay calls.
2. **Synthesize**: for answer-only approaches, an `.nl` author writes TypeScript from (input, output) pairs plus the
   instructions. Replay verifies the result.
3. **Improve**: the author may write a better body than the recorded behavior (merging two equivalent approaches,
   removing a wasted eval, fixing an inconsistency). Recorded outputs are then not the reference (6.1).

A body may throw `Deopt` before its first effect to decline an input the guard admitted. This is the side exit; the
runtime then runs the model. Once a body has caused an effect it may not deopt. A later failure is a call failure,
exactly as for a model execution. Folder transactions make it possible to deopt from directory reducers as well (the
transaction is aborted), but v1 leaves reducers out.

## 6. Acceptance and lifecycle

### 6.1 Two acceptance modes

- **Faithful**: on a held-out split of recorded calls the guard admits, the body's output equals the recorded output
  under the return type's equality (exact for records, enums and numbers; for free-text fields, an equivalence
  judge whose verdicts are themselves sampled for review). The executor was nondeterministic, so the reference is the
  *agreeing majority* of recorded outputs for equal inputs. Inputs whose recorded outputs disagree are excluded and
  reported. Disagreement is a finding about the program.
- **Improving**: the body is scored on the program's evaluation suite and on annotated calls (3.6) against the model
  fallback. Acceptance follows the adaptation system's rules: the baseline is always a candidate, the scorer is
  independent, the comparison is paired. An improving specialization changes behavior and is reviewed like a source
  patch.

### 6.2 Tiers

```
observed → candidate (verified offline) → shadow → active → (demoted)
```

- **Shadow**: the body runs next to the model on guarded calls. It runs with no effects: services are replaced by the
  replay harness's recorded results from the model run. The model's result is still returned. Divergences are
  recorded.
- **Active**: the body's result is returned. A sampled audit (a declared fraction of guarded calls) also runs the
  model in the background and compares results. A divergence rate above the acceptance bound demotes the
  specialization automatically, and the demotion is recorded.
- A specialization is bound to the definition revision, the build, and the executor identity whose traces it was
  learned from. A change to any of them puts it back in shadow. This mirrors adaptation compatibility checks and
  rewrite gating.

### 6.3 Artifact and source forms

- **Artifact** (runtime overlay): a new adaptation component kind, `specialization`, carrying guard source, body
  source, feature extractors, acceptance mode, evidence (call ids, splits, precision/coverage, divergence counts) and
  compatibility. It is activated, rolled back and exported through the existing adaptation lifecycle
  (`adaptation/compatibility.ts`, `optimization/promotion.ts`, `export-patch.ts`).
- **Source** (pluggable hot path): export as a crisp implementation next to the `.nl` file, selected by a setting,
  following the owner's pluggable-hot-paths rule. The guard stays explicit in source so a reader sees when the crisp
  path applies.

### 6.4 Runtime dispatch

In `runDefinition`, before the agent is built: look up `(definition revision, site template)` in the task's active
specialization table, evaluate guards in order, and run the first body that admits. Body calls get a trace of their
own (`specialization` events, `invocation` with `executor: { kind: 'crisp', specialization_id }`), so specialized
calls are recorded in the same store and compared with model calls. Nested natlang calls from a body go through the
kernel as usual and can be specialized in turn.

Nothing changes on the model-facing surface. Executing models see no new names, tools or prompt text. Specialization
is host and runtime behavior (language-design-restraint).

## 7. The specializer application

```
applications/specializer/
  specialize.nl                 top level: profile, pick targets, run the per-definition loop, report
  specialize/
    targets.nl                  which hot definitions are worth it (cost × volume × apparent regularity)
    sameApproach.nl             decision: do two eval programs do the same thing
    proposeGuard.nl             examples of one cluster vs the rest → a candidate predicate in TypeScript
    writeBody.nl                template / examples / instructions → a TypeScript body (lift, synthesize, improve)
    explain.nl                  a reviewer-facing description of each specialization and its evidence
  index.ts                      crisp: normalization, anti-unification, features, rule induction,
                                precision/coverage, replay verification, splits, artifact assembly
```

It uses the program-improver's `iterateOn` reducers and evaluation capability, and the `traces` service (3.5). It
returns specialization artifacts and a report, and never activates anything itself. Activation stays an explicit
operation, as for adaptation artifacts.

Per the port-granularity rule, each of these parts gets an explicit decision (NL function / instruction / crisp
helper) in a short DECOMPOSITION.md before implementation.

## 8. Order of work

1. Call record store: row schema, blobs, `callStoreSink`, exact capture by default, `parentActionSeq`, inline
   template identity, cost rollup. Default in the CLI and application runner. Tests: round trip, dedupe, parent-action
   links, opt-out.
2. `traces` service and `natlang traces` CLI. Migrate `iterate` statistics and the improver's evidence reads.
3. Replay harness with recorded-effect serving and divergence reports.
4. Runtime dispatch, the `Deopt` side exit, shadow and audit modes, `specialization` adaptation component kind and
   schema.
5. Specializer application: crisp mining first (normalization, anti-unification, features, rule induction), then the
   `.nl` judges and authors.
6. First targets: pick high-volume definitions from recorded application runs. Likely candidates are nldb's
   `classify`/`filter/meets`, pi's per-message policies and the compilers' pattern-heavy passes. Report faithful
   coverage, divergence and saved model time per target.
7. Skills and docs: `skills/natlang-integration` gains recording, `traces` and specialization activation. Authoring
   skills change only if authors gain something to write, such as a recording opt-out.

## 9. Decisions for the owner

1. **Exact recording by default.** Section 3.2 makes exact value capture the default with a per-program opt-out.
   Alternative: opt in per program. The default is what makes specialization (and improver evidence) work without
   setup.
2. **Store location and format.** Per-application SQLite plus content-addressed blobs under the state directory,
   versus one machine-wide store. Per application is simpler and matches today's `state/traces`.
3. **Improving mode in v1?** Faithful-only is cheaper to trust. Improving mode needs evaluation suites or annotations
   per target program.
4. **Effects.** v1 specializes only definitions whose recorded calls are pure or use services only. Folder reducers and
   capture writes come later.
5. **Output form.** Runtime artifact first and source export second, or the reverse.
6. **Training use.** Specialized calls stop producing model trajectories for their guarded region. Should a sampled
   fraction keep running on the model so training data for those patterns keeps growing, and should guarded-region
   recordings be marked as "crisp-solvable" for curriculum weighting?

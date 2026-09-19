# Infrastructure implementation plan for the complete project portfolio

Planning revision: 2026-09-20. Status: concrete proposed implementation sequence; no work below is reported as implemented. All 20 project families are in scope. This document is the sequencing authority where earlier roadmap/refactor task ordering differs. The [execution interfaces](EXECUTION_INTERFACES.md) supply interface design; the [project index](projects/README.md) supplies product requirements and C0–C7 capability definitions.

## 1. What to optimise

Implement the smallest shared capabilities that remove actual blockers across several projects. Prefer a reusable implementation boundary over a new model operation. A capability is complete when a consuming natlang programme runs against it and its contract has been checked, not when an interface class exists.

There are three kinds of work:

- **Core/interface work:** exact value boundaries, invocation identity/configuration, engine selection, checked programme loading, observation records and existing combinator semantics.
- **Optional host work:** shared JS environments, SQL, processes, files, search, model help, rendering, storage and recovery. An embedding supplies only its programme's dependencies.
- **Programme/library work:** dependency resolution, cache policy, semantic merging, type inference, repair strategies, incident analysis and UI behavior. These remain natlang plus crisp library code.

Keeping the full portfolio in scope justifies investing in common seams and traces early. It does not justify making every target import every host service.

## 2. Priority by concrete reuse

Priority is qualitative: direct consumers and prerequisites, not an invented performance score. Many features improve all projects indirectly; that is not counted as a direct unlock.

| Order | Work package | Direct blockers removed | Why it is early / why it stops |
|---|---|---|---|
| I0 | Baseline and compatibility contract | All subsequent changes | Prevent accidental semantic/training changes; a small fixture baseline, not a complete spec rewrite |
| I1 | Invocation context, model settings and seeds | P02, P08, P09 experiments, P16, P17; identity prerequisite for traces | Small API change with high cross-cutting value; no scheduler |
| I2 | Engine seam and common value conversion | P01, P03, P04, P10, P13 and other host-backed programmes | Removes direct QuickJS coupling; preserve current execution first |
| I3 | Source loading and checked meta-APIs | P05, P06, P07, P10, P11, P16, P17, P20 | One source/run boundary supports most developer tools |
| I4 | Structured reduction trace and reader | P06, P16, P17; teacher collection and debugging throughout | Build observation once, before several engines create incompatible logs |
| I5 | Versioned engine selector and useful host implementations | P01, P03, P04, P10, P12, P13, P14, P18, P20 | Turns I2 into real capability; native sharing remains optional |
| I6 | Stream Fold and long-work event integration | P04, P06, P08, P11, P12, P15, P19 | Largest remaining interaction blocker; no ambient interrupts |
| I7 | Scenario/replay/teacher integration | Training for all projects | Reuse existing IR and validate whole runs; do not collect a large obsolete-surface corpus |
| I8 | Bounded independent Map and stream composition | Throughput for P01, P03, P08, P10, P16, P17 | Valuable once ownership/seeds/traces are settled; not a prerequisite for serial applications |
| I9 | Second embedding and common developer experience | Browser P06/P11 and portability validation | Test real portability before freezing assumptions into mature tooling |
| I10 | Optional persistence and recovery host library | Full P03/P04/P06/P11/P12/P15/P19 scopes | Implement where restart promises require it; keep out of lightweight targets |

This is not a single chain. I2 and I3 can proceed after I0. I4 needs I1's identity convention, but its schema can be designed alongside I2/I3. I5 needs I2. I6 builds on invocation/evaluator boundaries and a minimum trace. I7 starts as soon as I3/I4 and one real adapter are available. I8/I9/I10 each have independent consumers once the common interfaces are stable.

No sub-agents are required or assumed by this plan. Independent work packages describe dependencies, not a delegation instruction.

## 3. Target interfaces and placement

Proposed names are deliberately small and provisional. Add modules only when extracting real code; do not precreate a framework hierarchy.

```text
natlang/invocation.py       Run options, logical call identity, request settings
natlang/execution.py        Crisp request/result contract and engine bindings
natlang/trace.py            Structured events, sink contract, value views
natlang/streams.py          Host-source contract and bounded consumption
natlang/codebase.py         Checked definitions/linking, independent of file access
natlang/host.py             Convenient file/stream bindings and embedding entry points
natlang/js.py               Existing JS/TS adapter, compatibility implementation
hosts/                     Optional implementations and native application libraries
scripts/                   Trace inspection, scenario collection, materialisation
```

Keep one TS-style natlang type system. Keep source syntax stable except explicitly versioned engine metadata/tool changes. Keep the interpreter's action inventory stable; the proposed new required `engine` argument belongs on `run_code`, not a separate tool per engine.

A run supplies: checked programme + typed inputs + model driver/options + available engine/environment bindings + seed policy + optional trace sink. Host access inside eval uses convenient native APIs. Do not require all native objects to implement a universal remote-object protocol.

## 4. Work packages

### I0 — Freeze the behavior we must preserve

**Touchpoints:** `spec/SPEC.md`, `natlang/surface.py`, `runtime.py`, `types.py`, `values.py`; current conformance and canonical trajectory tests.

Capture representative cases for typed writes and rejected writes, missing versus Null/empty values, nested calls, finite Map/Fold/Iterate, quiescence/resume, completion/marks, crisp exceptions and effects before failure. Record current model/tool/profile identities. Identify existing spec drift and choose which discrepancies to fix separately from refactoring.

**Deliverable:** a compact compatibility manifest and fixture pack with expected typed outcomes, action outcomes and effect order. Run existing relevant tests before extraction. Use semantic/action comparisons rather than Python object addresses or timings.

**Gate:** every subsequent refactor can compare against this baseline; deliberate differences are versioned. Existing working-tree teacher/IR work is reviewed/preserved rather than included accidentally in an infrastructure patch.

**Stop:** do not rewrite all documentation or repair unrelated behavior before I1–I3 can begin.

### I1 — Explicit invocation context and randomness

**Touchpoints:** `Runtime.__init__`, `_run_episode`, `_instantiate`, `ToolAgent.run`, `decoder.py`, `native.py`, CLI/probe configuration. Start in `tests/test_model_agent.py`, `test_native.py`, `test_agent_support.py`; add focused invocation fixtures.

1. Define the model-turn protocol actually consumed by `ToolAgent`, preserving the existing `ChatTurn` where suitable. Raw formatting/generation stay additional backend interfaces.
2. Introduce run options and per-invocation settings. Stop sharing mutable request deadlines/seed state between unrelated invocations. Preserve current defaults through an explicit compatibility constructor/profile.
3. Assign stable logical call/attempt identities independent of Python `id()` and completion order. Keep run identity separate from the reproducible call path.
4. Add explicit root seed policy and named random purposes. Freeze derivation encoding/version and test vectors before claiming cross-platform consistency. Record backend range conversion and support. Replace hidden `seed=0` calls in the selected new profile, including review calls.
5. Keep host/world RNG separate from model sampling. No attempt to promise that ambient native randomness is controlled automatically.

**Demonstration:** P02 merges the same supplied history in two independent runs with the same profile; P09 executes a finite spell fixture. Use recorded/fake drivers for exact plumbing tests and actual model runs to measure reproducibility.

**Gate:** call-seed vectors are invariant under unrelated independent completion order; explicit resampling changes the named attempt; unchanged compatibility settings preserve baseline behavior. Budgets cannot be accidentally reset by nested calls.

### I2 — Extract crisp execution and define value boundaries

**Touchpoints:** `Runtime._run_crisp`, `Session._do_eval`, `_fx`, `js.py`, `js_worker.py`, `prelude.js`, `values.coerce`, evaluator-related tests in `tests/test_surface.py` and `test_review_fixes.py`.

Inject a minimal executor binding. Route authored crisp functions and inline snippets through it, preserving their different execution modes. Move JS-specific conversion, preparation and errors into the JS adapter. Keep typed state commit and validation in the runtime. Engine failure cannot imply effects were undone.

Write shared conversion fixtures: missing versus Null; optional fields; booleans; finite numbers and exactly representable integers; nested records/lists; unrepresentable binary/native values; disposed host references. Reject unsupported/lossy conversions explicitly. Do not introduce full TypeScript or opaque host types in this package.

**Demonstration:** run existing crisp examples unchanged through the injected adapter and a recording substitute. The substitute proves the boundary, not language compatibility.

**Gate:** existing JS/TS results, effects and diagnostics remain equivalent under compatibility mode; a malformed adapter result is rejected by the common validator. Importing the core does not initialise optional database/process/rendering services.

**Stop:** no multi-language feature or engine-selection surface yet. This patch should isolate changes in behavior from changes in implementation.

### I3 — Checked source graph and eval-accessible meta-operations

**Touchpoints:** `codebase.py` (`FunctionDef`, `load_function`, `from_inline`, `check`), `host.py` (`load`, `instantiate`), `values.load_program`; `tests/test_codebase.py`, `test_program_ir.py`.

Extract validation/linking over already supplied definitions. Retain filesystem/frontmatter loading as an adapter; add explicit in-memory source/value binding. Do not infer filesystem access from ordinary embedded text values.

Expose a small embedding API for load/check/invoke and read-only definitions/type operations. An eval environment may supply wrappers for these functions. Child runs receive selected inputs, engine bindings, model settings, budgets and trace ancestry. They do not inherit the caller's private locals or all host authority implicitly. Source revisions remain immutable for active runs.

**Demonstration:** P05 checks an in-memory example; a P10 fixture loads and invokes a newly edited cell; P07 validates a supplied dependency tree. Use only the exact operations each requires.

**Gate:** file, inline and in-memory frontends create equivalent checked function graphs; missing/cyclic links fail before execution; editing the source container does not mutate a running definition; a child run has an explicit parent link and bounded work.

**Stop:** no package registry, hot code replacement, new function creation tool or universal module format. Existing IR/source representations come first.

### I4 — Reduction trace data and a small reader

**Touchpoints:** `Session.apply`/`act`, `_swap_out`, `_quiesce`, combinator transitions, invocation/evaluator boundaries, `nodes.py`, `values.dump_state`, teacher capture/materialisation scripts. Current `test_canonical_traces.py` and `test_teacher_trajectory_ir.py` provide useful compatibility context.

Implement the [trace proposal](EXECUTION_INTERFACES.md#7-execution-and-reduction-traces-as-data) in two increments:

- **I4a:** versioned manifest, logical IDs, node/action/eval lifecycle records, optional in-memory/JSONL sinks and a small read/query library. Capture structured calls/arguments and outcomes; do not parse human-readable log text to recover execution.
- **I4b:** typed reduction changes with sufficient initial state to reconstruct tree values, pending nodes, progress and provenance. Add recorded-decision replay using supplied observations. Mark missing native state/effects explicitly.

A sequential host uses one producer; parallel hosts can later attach local sequence numbers and causal links. Do not require a global event store. A shared native object may be shown as an opaque descriptor; the reader must still open the trace and explain reconstruction limits.

**Demonstration:** a command-line inspector shows a nested call, a rejected write, a Fold step and an effect followed by failure. Opening it executes nothing. This is the first P06 debugger component and a P17 input.

**Gate:** reconstructed final typed state matches execution for the supported fixtures; proposed/rejected actions cannot appear as applied; replay has no live effects; missing capture is visibly incomplete. The trace viewer distinguishes replay from live rerun.

**Stop:** no complete IDE, distributed tracing service or arbitrary native-memory checkpointing.

### I5 — Engine selection, shared host access and high-reuse adapters

**Touchpoints:** `surface.py` run_code schema, `_op_run_code`, authored-function loader metadata, `native.py` grammar, decoder adapters, prompts, reference generators, conformance fixtures, IR projection/export scripts. Optional host implementations live outside mandatory core imports.

**I5a — Surface version.** Require `engine` on the new `run_code` surface, constrained to available bindings. Authored crisp functions bind their engine at load time; ordinary `call` stays unchanged. Map historical engine-less traces to their known old engine through an explicit projection; preserve originals. Test one-engine and multiple-engine cases. No silent fallback.

**I5b — Shared TS environment.** Retain the current isolated JS/TS adapter. Add a retained JS-host implementation where code can operate directly on native application objects. An initial desktop implementation may be a persistent JS worker controlled by the Python runner: the transport passes eval requests/results, while jobs/buffers/objects stay native inside that worker. This is direct sharing with the JS host, not direct sharing with Python and not a proxy protocol for every object. A future JS embedding can run the same host library in process.

Specify fresh/retained lifetime, disposal, source mode and how host waits settle. The host binding owns resource limits, actual enforcement and cancellation support. Keep natlang-owned state behind validation. Trace enough identity/settings to distinguish isolated from shared execution; do not claim shared access is sandboxed.

**I5c — Useful optional libraries, in this order:**

| Library/adapter | Minimal operations | Consuming demonstrations |
|---|---|---|
| Files/processes/native bytes | Read snapshots; execute argv; inspect bounded output; retain native jobs; explicit cancel/outcome | One P01 render and one P03 build, then P04 |
| SQL/data access | Bound query/statement execution; result conversion; scoped connection/view; transaction outcomes | P10 joins two tables; P13 previews/applies a patch |
| Search over supplied data | Exact scan/lookup first; optional indexed implementation behind the same local API | P14 finds evidence; P05/P20 inspect source |
| Model assistance | Explicit request/config, returned typed observation, usage and failure | P01 frame inspection; P12 difficult evidence |
| Rendering | Render a typed view/document and expose host events | P18 two outputs; P06 inspector view |

These are ordinary APIs inside eval, not core tools. Process execution through argv avoids making a Bash engine a prerequisite. Add Bash explicitly when terminal recipes justify it. A model-assistance library must not hide the entire application workflow in a stronger model.

**Gate:** a native job/buffer survives between evals in one declared environment and becomes invalid when that environment ends; two runs do not accidentally share state; unsupported engine/value conversion fails explicitly; selected native mutation is visible in a subsequent eval and trace coverage is honest. One-engine student behavior is measured before broad multi-engine collection.

### I6 — Stream Fold and long-running operations

**Touchpoints:** `OpenList`, `Runtime._run_fold`, `host.load_fold`, pending-node handling, source integration in `scripts/serve_web.py`; stream/webserver fixtures. Preserve existing finite lists and their addressing.

Define a small host-source result: item available, temporarily empty, closed, or failed. Replace reliance on a possible data string such as `$close` in the new boundary; keep compatibility conversion explicit. Record consumed position separately from buffer index.

Implement a stream-fold controller retaining accumulator, current step and source position. Waiting for input is host suspension, not a model turn, a blocker or normal stream closure. The synchronous runner may wait; event-loop embeddings need a nonblocking driving path. Begin by driving complete function invocations at event boundaries. Introduce internal continuation support only where a model/eval call must yield to that host; it is not a new model-facing coroutine language.

Bound unread buffering and release consumed payloads according to the stream contract. Do not silently trim a finite list or reuse an index for another item. Define historical access through captured trace/source storage when available; absent history is explicit. Long jobs live in eval; their start/inspection/cancel APIs produce ordinary summaries and completion events to the application stream.

**Demonstration:** P04 starts a process, consumes another user event, then receives the completion; P12 consumes windows without retaining every log in the tree. A small P19 replanner handles a changed constraint as the next event.

**Gate:** temporary emptiness never completes the Fold; closure drains admitted work; failure is distinct from closure; each Fold step finishes before the next mutates state; late job results keep original identity; queues and retained payloads stay bounded over a long fixture. Cancellation does not imply external rollback.

**Stop:** no interrupts inside a lambda episode. Exact pause/cancel support of a host operation remains adapter-specific. Do not turn a waited-on source into a general message bus.

### I7 — Whole-program scenarios, replay and teacher collection

**Touchpoints:** `scripts/program_ir.py`, existing program/scenario IR builders, `teacher_trajectory_ir.py`, `materialize_teacher_trajectory_ir.py`, `export_sft.py`, manifests and canonical-trace tests.

Link execution traces with existing semantic programme IR and teacher decisions. Extend collection beyond leaves to root/child runs, supplied engine bindings, stream items and host observations. Preserve rejected, proposed-but-not-executed and failed actions. State clearly which native effects cannot be reconstructed.

Create resettable fake environments implementing the consumed contracts, plus small real integration fixtures. Add a common scenario runner with source/input/config hashes and paired model comparison. Expose it through eval for P16/P17; reuse CLI tools rather than creating a second training pipeline.

**Demonstration:** a teacher executes P01 or P03 and a stream application. Replay checks typed reductions, declared outcomes and relevant effect sequence; accepted episodes export through the student's current template. A failed/unknown outcome can be correct if the scenario contract requires it.

**Gate:** no provisional or unlinked evidence is silently treated as gold; train/dev/test keep complete programme/contrast families together; engine/seed/source changes produce new manifests; no live effect during replay. Run student pilots before scaling collection. Shared-state cases enter replay training only with sufficient recorded observations/reset semantics.

### I8 — Parallel Map and bounded stream composition

**Touchpoints:** `_run_map`, invocation context ownership, decoder usage/deadlines, result commits, stream adapters and causal trace links.

First implement bounded concurrency for finite independent Map slots. Isolate child runtime context/random streams and give each result a stable slot identity. Aggregate budgets explicitly; preserve index order and completed/quiesced slots on resume. Shared mutable native environments default to serial use unless their contract supports the particular parallel access.

**Demonstration:** P16 runs independent trials and P03 executes a verified ready batch. Compare serial and parallel exact fixture outputs; actual model equality is required only under a profile where it is promised. Test causality and result association separately from numerical reproducibility.

For unbounded input, first compose host stream windows with ordinary finite Map and forward ordered outputs. Label this honestly as windowed composition, not an implementation of a final infinite `B[]` result. If direct streaming Map is needed, freeze its incremental output, closure, retention and historical-address semantics before changing the pending-node representation. This representation decision is still open in the interface spec and is not allowed to leak into an accidental API.

**Gate:** admission/buffering are bounded, one slow slot cannot cause unlimited growth, completion order cannot reassign randomness or values, failed slots resume correctly, and single-threaded serial behavior remains fully supported.

### I9 — Second embedding and consistent developer experience

This is a real compatibility project, not just another Python adapter. Start with in-memory definitions, recorded model decisions, portable values and the same trace fixtures in a lightweight JS/browser embedding. Implement only the semantics required by the fixtures, then expand to shared conformance before claiming general language compatibility.

Bind browser-native eval/rendering explicitly. Initially the model driver can be remote; local inference adds actual model/operator/memory/latency validation. Use P06's inspector and P11's one-page/one-cell example as consumers. A common trace reader must handle both hosts' standard events and opaque native descriptors.

**Gate:** cross-host type conversion and seed vectors agree; declared compatible scenarios have equivalent typed reductions/outcomes; unsupported engines/source forms are explicit; active source revisions remain immutable. Browser replication must independently meet P02's model-profile reproducibility requirement; language compatibility alone does not establish it.

**Stop:** no requirement that the browser implement shell execution or every desktop library. File packaging is an adapter, not part of the language port.

### I10 — Optional durable hosting and recovery

Build this only after actual workflow/stream contracts exist, but keep it in scope for the promised full portfolio. Share durable primitives among consumers once their requirements overlap.

Persist application state at completed Fold boundaries, consumed source position, pending operation identities and receipts. Use P15 to exercise lost acknowledgements and compensation; P12 exercises notification deduplication; P04/P11 exercise session/revision restoration. Start with one local persistence implementation rather than a storage abstraction catalogue.

Persisting an application checkpoint is not equivalent to resuming an arbitrary live eval/native object. Recreate native clients/jobs from their host contract or retain explicit uncertainty. Add suspended-episode snapshots only if step-boundary restart cannot meet a concrete requirement; full interpreter state and model context then need a separate snapshot specification.

**Gate:** crash injection before dispatch, after external commit and before local acknowledgement preserves obligations and avoids unjustified duplicate effects. Exactly-once external behavior is claimed only where the adapter actually supports it. Lightweight embeddings remain usable without this module.

## 5. Proposed patch sequence

Each row is a reviewable scope; large rows may be split while retaining the gate. Effort is relative, not a calendar estimate. Interfaces/source manifests make semantic changes visible.

| Patch | Scope | Depends on | Effort | Required evidence |
|---|---|---|---|---|
| 01 | I0 compatibility fixtures and deliberate discrepancy list | — | Small | Baseline conformance/action/effect outcomes |
| 02 | Model-turn protocol and request configuration ownership | 01 | Medium | Existing backend and recorded-driver paths |
| 03 | Logical invocation identity and explicit seed policy | 02 | Medium | Seed vectors, nested/review invocation tests |
| 04 | Inject existing crisp executor; isolate conversion/errors | 01 | Medium | Existing JS/TS behavior unchanged |
| 05 | Checked source/value loading independent of files | 01 | Medium | Equivalent frontends; immutable active source |
| 06 | Trace manifest/lifecycle/actions/eval records and reader | 03, 04 | Medium | No code execution on inspection; structured evidence |
| 07 | Typed reduction changes and sequential reconstruction | 05, 06 | Larger | Nested calls/combinators/failures reconstruct |
| 08 | New engine-selector surface and explicit old-trace projection | 04, 06 | Medium | Grammar/prompt/reference compatibility; one-engine probe |
| 09 | Retained shared JS environment + process/file helpers | 04, 08 | Larger | P01/P03 real fixtures; native lifetime and effect coverage |
| 10 | Eval-accessible source/type/run/trace meta-APIs | 05, 06 | Medium | P05 and one child-run notebook fixture |
| 11 | SQL binding + common conversion fixtures | 08 | Medium | P10/P13 typed query/transaction outcomes |
| 12 | Stream Fold source contract and host driving path | 03, 06, 09 for process demo | Larger | P04/P12 finite-to-live fixtures and bounded memory |
| 13 | Whole-run scenario/teacher capture and replay/export | 07, 08, 10; 12 for streams | Larger | Admitted trace-linked shard and paired student pilot |
| 14 | Optional search/model-help/rendering library extraction | Real consumers from 09–13 | Medium each | P14/P01/P18 demonstrations; no global tools |
| 15 | Bounded independent Map | 03, 06, 12 where streaming | Larger | P16/P03 serial/parallel contract checks |
| 16 | Second embedding + common reader/conformance subset | 05–08, stable interfaces | Larger | P06/P11 compatibility fixtures |
| 17 | Application checkpoint/receipt host library | 12, P15 service contracts | Larger | Crash-boundary recovery tests |

The first coding batch is patches **01–04**, followed by **05–06**. Those establish the seams and identity/trace base without waiting for a full host implementation. Finite P02/P09 programme work can begin alongside them. P01/P03 supply real consumers for patch 09; do not wait until patch 17 to build applications.

## 6. Release cuts and unlock evidence

“Unlock” means the necessary natlang/embedding capability is available, not that the product has been implemented.

| Cut | Included work | What should be demonstrably possible |
|---|---|---|
| A — Embedded interpreter | 01–06 | Supplied programme/driver/executor; controlled seed; inspectable basic run; finite P02/P09 experiments |
| B — Useful desktop host | 07–11 | Media/build commands with native objects; source/type meta-APIs; SQL notebook/data slice; reconstruction of captured natlang state |
| C — Eventful applications and training | 12–14 | Terminal/log/workflow streams; trace-linked teacher collection; search/inspection/rendering through eval |
| D — Scale and portability | 15–16 | Bounded parallel trials/build tasks; small second embedding and common trace reader |
| E — Reliable long-lived hosts | 17 + application contracts | Restart-safe supported workflows/sessions; unresolved remote outcomes handled explicitly |

Start actual project code at the earliest compatible cut. P07 and P20 can use local linking and basic process checks before a registry or complete build tool exists. P08 can use finite rounds before live streams. P11 can test semantic collaborative revisions before browser-local inference. This avoids artificial dependency cycles between applications.

## 7. Evaluation and migration discipline

Use targeted existing tests for each extracted boundary and add tests for semantic risks, not interface shape alone. For every changed model surface run a small paired teacher/student suite measuring wrong engine selection, malformed code, wrong bindings, false success and tokens/turns. A more general API is not automatically better for the weak interpreter.

Freeze source, semantic/tool version, engine/environment bindings, seed derivation and input contracts in each new trace build. Preserve old corpora and version projections explicitly. The exact names of new APIs/schema fields are frozen after their first consuming fixtures; do not churn them during large collection runs.

Review each patch for: new model concepts, mandatory dependencies, authority actually enforced, typed-boundary preservation, shared-state ownership, and truthful trace coverage. These are engineering acceptance questions, not extra user approval flows.

No core `Host<T>` type, universal artifact service, global search/delegate tool, distributed scheduler, hot-code mutation or in-episode interrupts are currently on the implementation path. Reopen one only if a project supplies a failing case that the planned interfaces cannot express well.

## 8. Concrete next action

Prepare patch 01's baseline and patch 02's model-turn/request-settings extraction, while sketching the executor request/result contract for patch 04. Then implement explicit seed/identity support and the executor seam. Keep application authors supplied with finite fixture interfaces throughout.

This plan intentionally invests first in execution boundaries, identity and observable reductions. The same small changes support many different applications; host libraries can then grow without enlarging the interpreter's vocabulary each time.

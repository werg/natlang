# Delivery roadmap: applications first, a small core throughout

Revised proposal, 2026-09-20. Read with the [minimal-core architecture](AMBITIOUS_ARCHITECTURE.md), [targeted refactor proposal](EMBEDDING_REFACTOR.md), and [20 project plans](AMBITIOUS_PROJECTS.md). This replaces the previous mandatory infrastructure sequence.

Selected interface work is specified in [EXECUTION_INTERFACES.md](EXECUTION_INTERFACES.md). This adds planned engine selection, shared-host evaluation, explicit seeds, portable traces and stream completion to the small-core direction. Host-type syntax remains an open proposal.

Detailed delivery contracts are now in [the individual project plans](projects/README.md). Their C0–C7 dependency matrix distinguishes prerequisites from optional later work and preserves a small host closure for each programme.

The [portfolio infrastructure implementation plan](INFRASTRUCTURE_IMPLEMENTATION.md) now defines the concrete cross-project priority, 17 proposed patches, dependencies and release cuts. Its sequence takes precedence over older numbered task lists; this roadmap retains product milestones and training policy.

## 1. Working method

Keep the user's original programme: author ambitious natlang applications, discover what they need, improve the system where justified, execute teachers, and train students from checked trajectories. Foundations now mean a clear semantic contract and a few replaceable implementation boundaries. They do not mean building a general job, persistence, artifact or scheduler platform first.

For each project, write a small complete `.nl` workflow with crisp helpers and the minimum host adapter. Run it against finite fixtures and then a real supported environment. Record limitations without precommitting to a core extension. Expand toward the full product after the first slice supplies evidence.

Each project owns its types, host requirements, scenario pack, acceptance checks and gap log. A project can use ordinary calls and in-memory state; durability and interactive background jobs become requirements only at the scope where the product promises them. Model-executed control flow must remain substantive.

## 2. Milestones

### M0. Clarify and decouple

Reconcile actual and intended behavior for completion, transport, marks, Map ordering and open streams. Characterise behavior before changing implementation. Apply the first small refactors in [EMBEDDING_REFACTOR.md](EMBEDDING_REFACTOR.md): injectable crisp execution, an accurate model-turn interface, and explicit source/value loading boundaries. Keep existing implementations and defaults.

These three refactors are included in the implementation plan. Execution-context ownership cleanup follows where needed for reentrancy or a concrete scheduling experiment. Refactoring preserves existing semantics; event/interrupt experiments are a separate decision.

**Gate:** existing relevant tests and canonical execution outcomes remain valid; no new model tools, syntax or types. An in-memory embedding can supply its programme and decision driver without a filesystem or model server. A fake executor tests the seam; it does not establish a second language's compatibility.

### M1. Three contrasting application slices

- **P02 semantic merge:** finite text histories, natlang merging, same model/shared seed, recorded outputs. No distributed service required to compare the two merge strategies.
- **P01 media:** inspect one fixture clip, select a transformation, call a narrow FFmpeg adapter, assess the result. Local files or ordinary references suffice initially.
- **P03 build:** build a small dependency graph, inspect a failure, propose one bounded repair, verify it. Start with serial execution and an application-local cache if useful.

**Gate:** each programme executes through the model using current mechanisms; failure behavior and domain results are independently checked. Any adapter timeout issue is resolved for that adapter, without disguising uncertain completion as a failed/no-effect operation. A long call may wait through the host's implementation; it does not automatically require a model-visible job protocol.

Run a small teacher pilot with success, ambiguity, recovery and unsupported cases. Evaluate the weak student on the same interfaces as soon as practical. A teacher-friendly design alone does not pass the ergonomics gate.

### M2. Test the embedding boundary and extract actual reuse

Exercise the same relevant contracts in a single-threaded host and, where worthwhile, a host that batches independent work. Concurrency is optional. Preserve call semantics and document the behavior of limits/randomness. Start with a single-threaded in-memory runner; a real browser is a later consumer, not a prerequisite for discovering obvious coupling.

Share media/build helpers only where the two applications actually agree. If both need managed processes, extract that adapter. If only one needs durable receipts, keep that facility with its host. Verify optional dependencies remain optional through their import/dependency closure.

**Gate:** the lightweight embedding imports and runs without the optional services; richer hosts satisfy their own contracts. No new general scheduling language is introduced merely to use more hardware.

**Selected interface track:** generalise crisp engine bindings and propose required engine selection on `run_code`; add explicit seed configuration; exercise isolated and shared-host environments; define execution/reduction traces as portable data. Complete incoming-event support as streams through Map/Fold. Version model-visible changes separately from behavior-preserving refactors, and run student interface probes before broad collection. In-episode interrupts remain deferred.

### M3. Grow the portfolio through local dependencies

Use the following order as a reuse guide, not a mandatory platform dependency chain:

| Cohort | Projects | Infrastructure supplied only where needed |
|---|---|---|
| Data and workspaces | P04 terminal, P07 packages, P10 notebook, P13 data studio, P14 evidence atlas | Processes, local source resolution, SQL or search per application |
| Analysis and tooling | P05 type tools, P06 IDE core, P17 specification explorer, P18 publisher, P20 repository migration | Source snapshots, trace capture, rendering and workspace operations |
| Reactive worlds | P08 games, P09 spells, P12 logs, P15 API workflows, P16 experiments, P19 scheduling | Host event loop; timers, queues and persistence according to the product |
| Collaborative/browser integration | P11 wiki, full P06 IDE | Tested semantic merge profile, browser rendering/inference and optional durable server |

Keep each first slice small. For a reactive prototype, `step(state, event)` can run over a supplied event list before connecting a live source. Build the IDE over functioning command-line workflows. Use the exact simulator before fine-tuning spells. Establish semantic merge behavior before relying on it in the wiki.

**Gate:** each application has a clear minimal host contract and separately identified richer product requirements. Do not make replay storage, a package manager or a vector database prerequisites for unrelated programmes.

### M4. Production scope and training transfer

Add durability, background interaction, more evaluators, distributed execution or browser inference when the chosen product scope requires them. Use the change-admission test before promoting any part into core semantics. Expand teacher collection only after fixtures, adapters and oracles have passed the pilot.

**Gate:** real runs support each advertised capability; optional host modules have their own failure tests; student evaluation includes original conformance and held-out application families. Publish autonomous versus assisted success and separate semantic quality from operational correctness.

## 3. Concrete implementation sequence

Follow [INFRASTRUCTURE_IMPLEMENTATION.md](INFRASTRUCTURE_IMPLEMENTATION.md) for source touchpoints, patch dependencies and acceptance gates. The sequence is based on assuming all 20 projects will be implemented while keeping host facilities optional.

| Release cut | Infrastructure delivered | First consuming work |
|---|---|---|
| A — Embedded interpreter | Baseline, model-turn/request configuration, seed/logical identity, executor seam, in-memory loading, basic traces | Finite semantic merge and spell fixtures |
| B — Useful desktop host | Reduction reconstruction, versioned engine selector, retained shared JS host, source/type/run meta-APIs, SQL | Media, builds, type tools, notebook/data slices |
| C — Eventful applications and training | Stream Fold, whole-program replay/teacher capture, extracted search/model-help/rendering libraries | Terminal, logs, workflows, evidence atlas, publishing |
| D — Scale and portability | Bounded independent Map, second embedding, common trace reader | Parallel experiments/builds and browser IDE/wiki slices |
| E — Reliable long-lived hosting | Optional application state/cursor/receipt persistence and recovery | Full workflow, session and incident recovery |

These are capability milestones, not a requirement to finish a platform before writing applications. Finite P02/P09 work starts alongside the first refactors. The first coding batch is proposed patches 01–04, then 05–06. A universal object protocol, artifact service, durable scheduler and opaque host-type extension remain outside the prerequisite path.

## 4. Questions and ownership register

These entries are investigations, not a list of approved features. “Needed by Pxx” does not mean “belongs in core.” Close an entry with a local solution when that meets the requirement.

| ID | Question | Default owner / smallest approach | Escalation evidence |
|---|---|---|---|
| G01 | Spec/tool/transport drift | Existing core conformance | Resolve intended behavior, not new abstraction |
| G02 | Serial versus parallel Map | Host execution strategy | Independent child isolation and semantic parity |
| G03 | Blocking streams/open Map | Complete host-source waiting/closure and Map/Fold semantics | Bounded memory, explicit source errors and incremental output/history rules |
| G04 | Durable receipts | Effectful application host | A product requirement for crash-safe recovery |
| G05 | Host callback outlives JS deadline | Specific evaluator/adapter | Preserve unknown outcome; decide waiting/job interface locally |
| G06 | Binary data | Keep native resources in crisp environment, expose IDs/summaries if needed | A real need to transport native values through the typed tree |
| G07 | Execution/reduction trace | Planned portable event schema and optional sinks | Cross-host readability, state reconstruction, explicit replay gaps |
| G08 | Multiple evaluators | Planned host engine bindings and versioned required selector | Consistent boundary validation; existing and shared-host adapter tests |
| G09 | Package/crisp dependencies | Loader and development tools | Real second source/evaluator target |
| G10 | Cache correctness | Build/notebook application | Explicit equivalence policy, no general runtime cache assumption |
| G11 | Coroutines | Ordinary calls and event-step programmes | Comparative weak-model evidence of expressive/ergonomic failure |
| G12 | Search state/history | Crisp environment operations first | No new global search tool; explicit host access contracts |
| G13 | Vector retrieval | Optional retrieval implementation | Measured recall benefit over simpler search |
| G14 | Stronger-model/vision help | Declared domain helper | A task requiring that capability; no global delegate tool |
| G15 | Shared-model/shared-seed reproduction | Planned explicit run seed and derived invocation settings | Cross-host derivation vectors and repeatability on supported backend |
| G16 | Semantic merge order/grouping | Natlang replication library | Compare histories and state/update results |
| G17 | Type inference quality | Development application | Checked obligations and held-out precision/recall |
| G18 | UI and interaction latency | View library and rendering host | Incremental scheduling demand; no core UI schema |
| G19 | Browser feasibility | Actual browser embedding | Required operator/memory/latency and transcript compatibility |
| G20 | Whole-application teacher replay | Training tooling | Preserve calls, outcomes and effects actually used by scenarios |
| G21 | Independent semantic grading | Project contracts/evaluation | Calibrated rubrics and exact checks where possible |
| G22 | Local/function limits | Programme decomposition first | Measured failure or excess complexity before raising limits |
| G23 | Source edits during runs | Editor/host immutable revision | Explicit migration only where needed |
| G24 | Work budgets | Existing limits + host request ownership | Shared accounting for hosts that spawn external work |
| G25 | Line marks | Existing semantic policy | Consistent conformance and honest training targets |
| G26 | Eventful execution | Streams to Map/Fold selected; defer episode interrupts | Evidence that stream composition is insufficient before considering another mechanism |
| G27 | Common type system and host escape hatch | One existing TS-style structural boundary; native values stay in eval initially | Shared conversion tests; opaque nominal syntax only for a demonstrated need |
| G28 | Shared eval environment | Explicit engine binding/lifetime/enforcement contract | Native state sharing works; trace accurately reports unobserved mutations |
| G29 | Meta-capabilities | Callable APIs inside crisp environment | Checked load/run/inspect with child-run scope and trace links |

Record the motivating scenario, classification, smallest alternative, model-learning cost, semantic interactions, porting burden and acceptance evidence. Classify model mistakes, ambiguous programmes and bad oracles before blaming the harness. A wrong branch is not automatically a reason to parse application prose in Python.

## 5. Scenario and contract format

Extend the existing semantic program/scenario IR rather than introducing an unrelated training source. These are development/collection records, not a required runtime event format. Omit host-specific fields when the scenario does not use them. An illustrative sidecar, not current accepted schema:

```yaml
schema: natlang.application_scenario/proposed
id: media-cancel-after-output
project: P01
source_revision: <content hash>
split_group: media-cancel-family-01
execution_profile: <model/runtime/prompt/seed manifest>
initial_world: <fixture manifest>
inputs: <typed values and artifact references>
event_schedule: <logical events and fault points>
grants: <scoped capabilities>
budgets: <turns, tokens, time, jobs, bytes, escalation>
expected:
  outcome: <success, waiting, blocker, error, or allowed alternatives>
  exact_contracts: <named executable checks>
  effect_contract: <counts, values and required causal ordering>
  semantic_rubric: <evidence-based acceptance criteria>
  forbidden_outcomes: <false success, lost data, changed requirements>
```

Use exact values where there is one correct result. Use predicates and rubrics for genuinely multiple valid outputs. Concurrent histories should be checked against required causal edges and per-resource order; a different valid interleaving need not fail an oracle written for sequential execution.

Every fake adapter should implement the same typed contract as its real counterpart, including the failure modes that counterpart exposes. A pure helper needs no simulated job or receipt service. Integration tests check that real adapters honour those contracts. Mock success is not evidence that a real subprocess, model, network service or browser works.

## 6. Teacher execution and training admission

### 6.1 Freeze the executable environment

For a collection batch, freeze source and transitive libraries, types, runtime, required evaluator versions, prompt/tool surface, relevant host configuration, fixtures, model/decoder settings, random profile, scenario contracts and split assignments. Capture the actual serving-model identity and exposed configuration. Hash binaries/model files where practical; a display name is insufficient.

Use the existing compact teacher setup as the starting configuration. Its documented small pilots do not establish reliability on larger programs. Run a preflight against tool transport, optional inputs, enum/record values, maps/folds, terminal behavior and failures before collecting application runs.

### 6.2 Run actual programmes

The teacher must choose actions while executing the `.nl` program through the same harness used by the student. Capture root and child episodes, tool schemas as offered, proposals, executed actions, outcomes, raw returned responses, observable reasoning fields when present, effect/job events and artifact identities. Distinguish actions proposed but never executed from failed attempts and successful commits.

Use resettable fake worlds for bulk fault scenarios and smaller real integrations for adapter validity. Retain every attempt, including budget exhaustion and rejected output. Run a pilot, fix identified issues, freeze a new version and recollect affected cases. Never silently repair an old transcript to make it look successful.

### 6.3 Audit with multiple independent checks

Admission requires all applicable layers:

1. Source and fixture hashes match the manifest.
2. Each executed action is valid under its actual offered surface and runtime.
3. Intermediate typed state, call binding and declared effects are checked.
4. Expected final state or declared failure outcome is satisfied.
5. Effect values, multiplicity and required ordering match the contract.
6. Semantic output satisfies an evidence-grounded rubric or adjudicated reference.
7. No unresolved artifact/model/source gaps prevent replay.
8. Split grouping, license/source permissions and provisional-gold rules pass.

Exact contracts cannot establish all semantic correctness. A stronger judge can help but is not infallible, particularly when it also generated the answer. Use blinded comparisons, different evidence presentations and human review of a calibrated sample, especially false-success cases. Replica agreement in P02 is an execution check, not a semantic-quality oracle.

### 6.4 Preserve three distinct durable layers

- **Program/scenario IR:** source meaning, fixtures, desired outcomes, contracts and provenance.
- **Teacher trajectory IR:** actual teacher decisions, attempts, responses, observations and audit links.
- **Materialised training view:** replay-verified turns adapted to the current harness and rendered with the student's template only at export.

Extend the existing whole-program reference machinery and teacher leaf materialiser carefully. Whole application replay needs the actual calls, state and observations of that application; include job receipts or event schedules only where used. Obsolete tool mappings must remain explicit versioned projections. Do not create successful action targets for actions that were never observed or justified by a verified reference execution.

### 6.5 Form useful training examples

Stream scenarios should vary arrival timing, temporary emptiness, source closure and independent completion order. Record the items actually consumed and the context visible at each step. Train valid continuation and preservation of effects without mixing ambient mid-episode messages into the existing semantics. If interrupts are explored later, keep their explicit experimental semantics and corpus separate.

A required engine argument changes the tool grammar and training view. Migrate old records through a documented projection that assigns the historical engine binding; retain originals. Train engine choice only over the engines actually available and report selection failures separately from code errors. Include shared-host state examples only under a declared environment profile; do not imply their native mutations are captured or replayable when they are not.

Keep short function episodes as the natural supervised unit, linked to the enclosing program and scenario family. Include control-flow execution, binding, exact-code use, evidence gathering, semantic judgments, valid repair, refusal to invent missing information, and correct error/blocker completion.

For failures, distinguish a correct failure report from a bad execution that happened to stop. Invalid actions can appear as past context for a verified recovery; they are not success targets. Store rejected trajectories for analysis or a deliberately designed contrast objective. Do not mix them blindly into ordinary SFT.

Separate the student executing its own workflow from examples where it appropriately requests a stronger model. Report autonomous success, assisted success, escalation rate and total cost. A student that forwards every task to the teacher has not learned to interpret it independently.

### 6.6 Split and evaluate

Keep entire programs, source projects, scenario/contrast families, paraphrase groups and derived leaf/review records together. Add explicit holds for application families, semantic task templates and larger scales. Deduplicate shared fixtures and generated variants across splits. Freeze dev/test before iterative teacher-driven repairs; retain a final untouched test set.

Evaluate:

| Dimension | Measures |
|---|---|
| Correctness | Whole-program success, exact contract pass, semantic rubric scores, false success |
| Honesty and recovery | Correct blockers/errors, unsupported requests, recovery success, duplicate effects |
| Cost | Tokens, model calls, help requests, wall time, process work, storage growth |
| Interactivity | Median/tail decision latency, queue delay, deadline misses |
| Reproducibility | Replay equality, restart parity, seed/profile agreement, merge disagreements |
| Generalisation | Held-out codebase/family performance, larger inputs, unfamiliar compositions |
| Stability | Original conformance suite and previous application regressions |

Use paired scenarios and identical evaluation manifests across checkpoints. Report uncertainty/sample sizes for empirical scores. Set project-specific semantic quality and latency targets after the first measured baseline, before the larger comparison. Require zero violations of exact invariants in the release suite; that is a tested gate, not a proof of universal correctness.

### 6.7 Fine-tune and promote

Build new immutable shards from admitted trajectories; preserve the previous corpus and manifests. Choose a measured mix of existing interpreter skills and new application episodes so a large verbose project cannot dominate by turn count. Compare balanced-by-program and balanced-by-family sampling. Start with a small training pilot and inspect actual action distributions, not only loss.

Run student-only and teacher-only evaluations, plus student-with-help where the application exposes that library. Compare compact state with any new state/scheduler surface before changing defaults. Promote a checkpoint only after affected application gates and original interpreter checks pass; retain the prior checkpoint and manifest for rollback. Spell-specialised models may coexist with the general interpreter, with explicit runtime identity in every run.

## 7. Admitting changes

Use the eight-question test in the architecture. Separate an implementation refactor from a new semantic obligation. Every proposal must name its layer, smallest alternative, weak-model burden and obligations for a small embedding. New syntax and tools need stronger evidence than a reusable helper.

Examples: fix a media adapter's timeout contract locally; put exact dependency validation in a build helper; keep merge policy in natlang; pass a document collection to a search function; implement a renderer for an IDE host. None alone justifies a new global operation.

Preserve simple defaults. A host may have richer capabilities without every programme discovering, selecting or managing them. When those differences affect correctness, express them in the relevant function contract and fail clearly on unsupported operations.

## 8. Completion at a declared scope

A project is complete when natlang executes its substantive workflow, the host implements the advertised operations, independent checks cover the promised outcomes/failures, and its collected examples have sufficient provenance and evaluation. A batch prototype need not claim background interaction; a memory-only game need not claim crash recovery.

The next deliverable is the targeted refactor and first finite semantic-merge, spell and media/build fixtures, expanded according to their individual dependencies. Stop expanding the foundation when those slices run; let their evidence determine the next changes.

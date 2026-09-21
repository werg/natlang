# Software that develops its own understanding

Status: proposed implementation plan, 2026-09-21. This document and its five
linked plans describe new work; they do not claim these capabilities are
implemented or that live-model evaluations have passed.

The objective is software in which natlang develops representations, algorithms
and interactions appropriate to the task. A successful application should be
able to acquire a useful capability that its author did not enumerate as an
action. It should then expose that capability as inspectable, executable code.

## The five directions

| Direction | Concrete application behavior | Detailed plan |
|---|---|---|
| Invent and retain abstractions | Turn a discovered method into a typed natlang function, test it and reuse it on another problem | [S01: Learned tools](projects/S01_LEARNED_TOOLS.md) |
| Revise the domain model | Discover that an existing category conflates distinct things, introduce a better representation and migrate its consumers | [S02: Revisable schemas](projects/S02_REVISABLE_SCHEMAS.md) |
| Invent interactions | Build a comparison, experiment or editor specifically for the uncertainty being investigated | [S03: Generated interactions](projects/S03_GENERATED_INTERACTIONS.md) |
| Preserve intent through change | Carry a requested change across code, data, tests and explanations; reconcile concurrent intentions semantically | [S04: Intent-preserving changes](projects/S04_INTENT_CHANGES.md) |
| Maintain explicit beliefs and evidence | Track why conclusions hold, what would defeat them, and what to investigate when observations change | [S05: Beliefs and investigation](projects/S05_BELIEFS.md) |

These are five interacting application capabilities, not five mandatory runtime
subsystems. Build them together in an executable research notebook, then reuse
the libraries in the existing applications.

## What is unnecessarily fixed today

The studio is a useful executable starting point, but its current architecture
still fixes too much of the application's meaning in JavaScript:

1. `studio/shared/contracts.mjs` enumerates all available domain actions.
2. `studio/shared/program.mjs` loads a fixed six-file program bundle.
3. `studio/studio.mjs` rejects an operation unless its supplied state equals the
   last host-produced state, and rejects a final reduction unless its entire
   state equals the final host-produced state.
4. `studio/shared/render.mjs` renders application-authored panels. Natlang's
   current view chooses headings, summaries, ordering and suggestions.
5. `StudioStore` persists state, effects, traces and native values, but does not
   yet provide a versioned workspace of generated source and related artifacts.

The whole-state equality condition is particularly consequential: it prevents
natlang from independently constructing an evolving semantic application state.
Removing it needs a precise replacement for effect verification, not another
host function that owns every domain update.

## Architectural stance

### Natlang owns the program

Natlang may construct state, introduce domain concepts, write functions,
generate crisp algorithms, select operations, inspect outcomes, revise its
approach and generate interactions. Exactness alone is not a reason to put an
algorithm in a fixed host implementation. A generated SQL analysis or TS
transformation is still part of a natlang-directed program.

Host libraries supply storage, engine execution, rendering, transactions,
process handles, model access and observations of actual effects. Keep a host
function crisp where its concrete responsibility requires it: atomically
changing a revision pointer, evaluating code, preserving a browser draft, or
reporting what a process actually did. The model cannot establish that an
external write succeeded merely by writing `success` into application state.

### Keep the core small

Use ordinary functions, typed records, source-as-data through the existing
loader/child-run seam, explicit evaluator bindings, and event streams. No new
`belief`, `schema`, `learn`, `transaction`, UI or interrupt language form is
proposed. Domain types use the existing structural TS-style subset; native
objects remain in the chosen eval environment.

Single-threaded hosts execute the same event sequence serially. Other hosts
may evaluate independent trials concurrently. Record logical invocation IDs
and consumed input order; scheduling is not part of the domain semantics.

### Persistence and continuation are different jobs

Continuation checkpoints already shorten the conversation inside an execution.
They are not durable application workspaces or semantic memories. Persist the
source, chosen representations, outstanding work and evidence needed to resume
an investigation separately from the model's continuation note.

Do not add a small run/turn cap. A long investigation can contain many small
typed calls and multiple continuation segments. Measure tokens per useful
result, repeated reads and repeated failed attempts. A program that is stuck
should expose its failure and revise its approach; a productive long run
should continue. Existing cancellation remains available.

## Shared changes, with concrete consumers

All names below are proposed library interfaces, not additional interpreter
tools or a universal protocol that every host must implement.

| Change | Concrete contract | First consumer and source touchpoints |
|---|---|---|
| Natlang-owned state | Reducer returns its own typed domain state; host independently verifies effect references, state version and declared exact invariants | S05; `studio/studio.mjs`, `shared/host.mjs`, browser application tests |
| Versioned workspace | Immutable artifact revisions, manifest, parent revision and atomic compare-and-swap of the active manifest | S05/S01; `shared/store.mjs`, new workspace library |
| Dynamic program bundles | Load a checked source closure from an immutable manifest, invoke a selected root with typed inputs, return result/trace/diagnostics | S01; `shared/program.mjs`, `shared/child-runner.mjs`, `shared/child-worker.mjs`, existing source loader |
| Queryable evidence | Retrieve selected objects, ranges and relationships with stable revision IDs; preserve complete originals in host storage | S05; extend `native_values` and evidence application |
| Generated view bindings | Versioned view tree/module, stable control IDs, typed handler bindings, revision-bound event dispatch and draft reconciliation | S03; existing `BrowserDomRenderer`, studio renderer and event dispatch |
| Candidate branches | Execute against an explicit artifact snapshot, keep results on that branch, activate an internally consistent manifest | S01/S02/S04; workspace library and existing child runs |
| Export/import | Bundle reachable source, state, evidence and recorded observations; identify native resources that must be rebound | All five; studio export and a new checked importer |

### Replace the state restriction carefully

1. Introduce a natlang-owned `domain` record and host-owned effect receipts.
   An event may change `domain` without any host operation.
2. An effect call takes operation inputs and revision preconditions. It returns
   an authoritative receipt plus observations, rather than forcing a complete
   application state replacement.
3. Natlang interprets the observation and builds the next domain state. At
   commit, validate the return type, workspace revision and referenced receipt
   identities. Verify relevant exact invariants against host observations.
4. Store event, domain revision, source manifest and effect references together.
   Do not pretend an IndexedDB commit is atomic with a Bash process or service.
5. If an effect completed before reduction failed, retain its receipt. A resumed
   run consumes that observation instead of issuing the effect again. Unknown
   outcomes remain unknown until inspected or reconciled.

Avoid replacing the fixed action menu with a generic `do anything` JSON tool.
Provide ordinary, documented host APIs through eval and generated typed
functions. Domain operations can evolve; storage and effect contracts remain
inspectable. Migrate one studio application first, then its actual consumers.

## Integrated product: an executable research notebook

Use Fieldnotes as the entry point, with Atlas evidence, Atelier source and
Counterexample tests available within the same workspace. This is one coherent
application, not four routes the user must manually coordinate.

First complete scenario:

1. Import two local observation tables and a methods note. Ask: “Did the change
   improve reliability, and for whom?”
2. Natlang explores the full data through crisp queries and records competing
   interpretations of ambiguous fields. It makes assumptions visible.
3. It discovers that a global comparison hides different device populations,
   writes a reusable cohort comparison function, and exercises it on examples.
4. It introduces an explicit observation/cohort representation, retaining raw
   inputs and unresolved device identities.
5. It generates a cohort comparison interface with controls for inclusion,
   provenance inspection and alternative assumptions. Controls emit events to
   natlang functions.
6. A late batch and a revised methods note arrive. Natlang revisits affected
   assumptions and conclusions, reruns appropriate computations and explains
   the changed answer without discarding the earlier result.
7. Two branches then differ: one corrects units, another changes cohort
   membership. Natlang merges their intentions, migrates affected source and
   explanations, and keeps any remaining ambiguity explicit.
8. Reload the browser and resume from stored workspace and pending work, without
   reconstructing the entire chat. Export and import into a clean workspace.

The test data must contain both a genuine confounder and irrelevant changes so
that blindly recomputing or inventing a confounder does not pass.

## Delivery order and gates

| Milestone | Deliverable | Gate before broadening |
|---|---|---|
| M0 | Natlang-owned state and revisioned artifact storage in one notebook workspace | A semantic state-only event commits; fabricated/stale effect references fail; reload preserves source and state |
| M1 | S05 evidence/assumptions and dependency-directed reconsideration | New contrary evidence revises the relevant conclusion; irrelevant evidence preserves it; full evidence remains inspectable |
| M2 | S01 function synthesis, execution and retained reuse | A generated function absent from the original action catalogue runs on a held-out case after reload |
| M3 | S03 generated interactions bound to those functions | A new task-specific control works, survives a view revision, and handles stale events/drafts correctly |
| M4 | S02 domain schema revision and consumer migration | A meaning-changing split of a concept migrates data, source and view on a branch without erasing unresolved records |
| M5 | S04 semantic change propagation and branch merge | Two independently useful changes survive a semantic merge; an incompatible pair produces an explicit question or alternative |
| M6 | Integrated notebook and reuse in two other apps | Full scenario above, clean import/resumption, and adaptation to logs plus wiki or data studio without a new runtime primitive |

Each milestone includes real UI, exact integration checks, teacher scenarios and
an explicit student-quality gate. Do not build all shared infrastructure before
the first end-to-end case. S01 and S03 can initially use a stable schema; S02 does
not block them. S04 can begin with fixed-schema edits before full migration.

## Evaluation and training

Maintain three separate forms of evidence:

- **Mechanics:** real engines, typed boundaries, storage conflicts, worker
  cancellation, stale events, import/export and effect recovery. Scripted model
  turns are appropriate here.
- **Semantic quality:** independent judgments of useful abstraction,
  representation fidelity, interaction usefulness, intent preservation and
  warranted conclusions. Passing type checks is insufficient.
- **Learning:** held-out transfer by the student, after training on reviewed
  teacher trajectories, including failure recovery and continuation segments.

Freeze each scenario's source manifest, complete inputs, model/tokenizer,
decoder configuration, seed derivation, evaluator identities and observations.
Same seed and model alone are not a cross-backend reproducibility guarantee.
Use controlled repeatability trials; preserve divergent results for inspection.

Train on complete algorithmic episodes with durable intermediate work and
short continuation segments. Include unsuccessful hypotheses and their useful
corrections. Admit samples only after checking their results and provenance;
a successful child run or a persuasive explanation is not an admission rule.

Hold out problem families and representation changes, not merely paraphrases.
Compare against the current fixed-action application and a no-retained-learning
variant. Measure completion quality, correction effort, unnecessary questions,
reuse, invalidated claims, elapsed time, tokens, context rereads and recovery
behavior. Use measurements to change implementations; do not mask weak results
with short episode limits.

## Runtime changes only if experiments demand them

Potential real blockers are: generated source closures that cannot be invoked
with existing loaders; useful types that the structural subset cannot express;
inability to recover an in-progress child invocation; and inability to publish
useful intermediate state during an open stream. For each, preserve a minimal
failing scenario and repair the existing seam where possible.

Initially use immutable source snapshots per invocation and normal queued
events. A newly learned function becomes available to a subsequent child run;
active lexical bindings do not change underneath a running reduction. This
keeps the lambda model understandable while allowing the application itself
to evolve over a long sequence of executions.

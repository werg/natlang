# Ambitious natlang projects: detailed catalogue

Status: proposed designs, revised against the settled execution interfaces, 2026-09-20. Read with the [minimal-core architecture](AMBITIOUS_ARCHITECTURE.md) and [delivery roadmap](AMBITIOUS_ROADMAP.md). Paths below are proposed unless explicitly identified as existing. Signatures describe intended interfaces; not every supporting type or capability is implemented yet.

There are **20 project families: the 12 original ideas plus 8 additions**. The general games family has three separately scoped prototypes. Each project starts with one end-to-end scenario and grows toward the full design. Natlang should execute the application algorithm; crisp code provides exact operations and host boundaries.

Cross-project infrastructure priorities, proposed patches and acceptance gates are in [INFRASTRUCTURE_IMPLEMENTATION.md](INFRASTRUCTURE_IMPLEMENTATION.md).

## Implementation plans

The [project implementation index](projects/README.md) maps all projects to concrete natlang capabilities, distinguishing existing mechanisms from planned runtime/embedding work. Each project now has a linked implementation plan covering typed boundaries, actual eval environment ownership, streams/reductions, delivery increments, failure tests and teacher/trace requirements. These plans refine the product scope below; their staged gates take precedence over earlier broad first-slice descriptions.

## How to read the host requirements

The detailed plans describe product ambitions. Their jobs, artifacts, search, UI, persistence and model-help facilities are **application or optional host facilities**, not proposed built-ins or a mandatory standard environment. All first slices should try the current tool/type/call mechanisms. The selected model-visible change is a proposed required engine argument on the existing eval tool; see [execution interfaces](EXECUTION_INTERFACES.md). Streams feed Map/Fold. Search and meta-operations live in crisp environments, which may share native host state. No separate global tools for those facilities are planned.

Native references should primarily live in the crisp environment; an ordinary typed ID can cross the natlang boundary when needed. A “reference” does not require a new primitive type; an “event” is an application record; an “actor” may be a state record stepped by the enclosing game loop. An ordinary domain call can wait for a host operation without making the model manage a job. Introduce visible job state only when the user-facing application actually needs background interaction or recovery.

The richer acceptance scenarios below apply as those features enter product scope. A synchronous first prototype is not required to simulate a durable service. Before promoting any missing affordance to the core, compare a local library/host solution and measure the weak student's ability to use it.

| Project | Minimum embedding for first slice | Add only for richer product scope |
|---|---|---|
| P01 media | Clip references, probe/render adapter, ordinary calls | Background progress, durable jobs, managed artifact store |
| P02 semantic merge | Text values, configured model and shared seed | Replication transport, checkpoint persistence, cross-device parity |
| P03 build | Local graph helpers, file/process operations | Remote cache, parallel workers, durable repair runs |
| P04 terminal | Session record and selected command recipes | Persistent sessions and interactive job control |
| P05 type tools | Supplied source snapshot and exact type checks | Project search index and incremental analysis service |
| P06 IDE | Source operations, renderer, existing trace observations | Durable time travel and training-job management |
| P07 packages | Local package records, exact resolver and file operations | Registry networking and publishing infrastructure |
| P08 games | World values, legal-action helpers, host game loop | Concurrent actor execution and persistent memories |
| P09 spells | Rule records and an exact simulator | Domain fine-tune and multiplayer integration |
| P10 notebook | Cell records, selected table/query helpers | Background cells, persistent data lineage and collaboration |
| P11 wiki | Page values, semantic merge library and renderer | Offline sync, local browser inference and durable server |
| P12 logs | Finite log windows and incident output | Streaming backpressure, persisted cursors and notifications |
| P13 data studio | Fixture tables and checked patch application | Live synchronisation and durable migration recovery |
| P14 evidence atlas | Passed documents and a simple search helper | Full-text/vector indexes and remote collections |
| P15 API workflow | Typed fake service operations and saga state | Real credentials, durable outbox and reconciliation |
| P16 experiments | World simulator, seeds and result records | Distributed experiments and persistent branches |
| P17 test explorer | Supplied programme, fixtures and checks | Broad schedule exploration and automated regression service |
| P18 publishing | Document values and two renderers | Publication service and richer binary formats |
| P19 scheduling | Calendar values and exact constraint helpers | Live calendar adapters and durable timers |
| P20 repository migration | Snapshot, patch operations and test runner | Larger workspace service and package-registry integration |

## Portfolio overview

| ID | Project | Main pressure on the system | First concrete result |
|---|---|---|---|
| P01 | Semantic media workbench | Binary artifacts, managed jobs, visual inspection | Verified transformation of a short fixture video |
| P02 | Semantic replication / notional CRDT laboratory | Seeded semantic merging, context agreement, history reduction | Replicas reproduce a merge from the same inputs |
| P03 | Build tool with bounded self-repair | Dependency resolution, caches, repair contracts | Incremental build with one verified repair |
| P04 | Semantic terminal | Composable recipes and persistent sessions | A multi-step workspace task with receipts |
| P05 | Natlang type inference/checking | Code retrieval and semantic call analysis | Evidence-backed signature suggestions and diagnostics |
| P06 | Natlang IDE | Generated UI, debugger, experiment management | Edit, run, inspect and replay one codebase |
| P07 | Natlang package manager | Locks, exact solving, compatibility evidence | Offline reproducible installation |
| P08 | Economic, fighting and NPC games | Actors, scheduling, bounded decisions | Three small deterministic-world prototypes |
| P09 | Spell programming game | Semantic compilation and domain fine-tuning | Free-form spell becomes a checked action plan |
| P10 | Natlang notebook | Cells, SQL, data lineage and invalidation | Reproducible mixed natlang/SQL analysis |
| P11 | Collaborative executable wiki | Semantic replication and browser runtime | Two users edit and run a versioned page |
| P12 | Log anomaly investigator | Streaming windows, evidence and escalation | Explain a seeded incident without duplicate alerts |
| P13 | Data reconciliation and migration studio — new | Ambiguous schemas and transactional patches | Auditable merge of two inconsistent datasets |
| P14 | Evidence notebook / research atlas — new | Search, provenance and conflicting claims | Answer with inspectable evidence and unknowns |
| P15 | Stateful API workflow composer — new | Protocol discovery, sagas and compensation | Recover an interrupted multi-service workflow |
| P16 | Simulation and experiment laboratory — new | Scenario branching and controlled randomness | Compare policies under matched simulations |
| P17 | Living specification and test explorer — new | Behavioral contracts and counterexamples | Discover and minimise an execution failure |
| P18 | Document compiler and publishing system — new | Semantic structure and multiple renderers | One structured source, two consistent outputs |
| P19 | Personal workflow / scheduling engine — new | Time, constraints and event reconciliation | Replan a disrupted schedule with explanations |
| P20 | Repository maintenance and migration agent — new | Source patches, build/test evidence and rollback | Verified migration in an isolated checkout |

## P01. Semantic media workbench

**Implementation plan:** [P01 — natlang prerequisites, embedding and delivery](projects/P01_MEDIA.md).

**Product.** A semantic interface to FFmpeg and related image/audio/video tools. It maintains a library of transformations, interprets requests such as “make a portrait version that keeps the speaker visible, remove the long pause, and make speech easier to hear,” inspects intermediate results, and decides whether further work is needed.

**Codebase.** `media_workbench/transform.nl(request, assets, policy) -> MediaResult`, with `understand.nl`, `choose_recipe.nl`, `plan.nl`, `inspect.nl`, `assess_result.nl`, and `revise.nl`. A recipe records intent, input requirements, parameter schema, transformation graph, test contracts, examples and tool versions. Reuse via `codebases/lib/media/`; keep each function's visible helpers within the existing listing limit.

**Execution.** Natlang identifies desired changes, checks available evidence, selects or adapts a recipe, requests a preview, assesses that preview, and either publishes the result or makes a bounded revision. Crisp functions probe streams, compile a validated filtergraph, select exact timestamps, build argument arrays, and compare technical properties. A process adapter executes FFmpeg in a scratch workspace. Outputs become artifacts only after validation. FFmpeg's own filtergraph language remains an exact embedded representation; escaping and compilation need dedicated helpers. [FFmpeg documentation](https://ffmpeg.org/ffmpeg.html), [filter syntax](https://ffmpeg.org/ffmpeg-filters.html).

**Inspection.** Choose representative frames, contact sheets and short audio segments; preserve selection provenance. Technical checks cannot establish that a crop retains the right person, so visual judgments use a capable inspection model when available. Unsupported modalities produce an explicit limitation, not a fabricated pass. Custom filter proposals use the same parse, preview, execute and verify route as library recipes.

**First slice.** Trim, crop and transcode a synthetic ten-second clip with a visible moving subject. Then add speech processing, multi-clip edits, and reusable semantic recipes.

**Scenarios and acceptance.** Missing audio, rotated inputs, variable frame rates, corrupt media, impossible target constraints, cancellation, process failure, output overwrite conflicts, uncertain completion. Check duration/resolution/stream properties, exact output provenance and no partial publication; score visual intent separately. Teacher trajectories should include useful inspection, bounded revision and honest unsupported requests.

**Affordances exposed.** Artifacts, long jobs, progress events, toolchain identity, multimodal delegation, a precise boundary between technical verification and perceptual assessment.

## P02. Semantic replication and notional CRDT laboratory

**Implementation plan:** [P02 — natlang prerequisites, embedding and delivery](projects/P02_SEMANTIC_MERGE.md).

**Product.** Explore “CRDT-like” collaboration where **natlang performs the merging semantically**, including interpreting intent, combining edits, preserving nuance, and resolving apparent conflicts. This is deliberately not a conventional CRDT with crisp merge algebra. Crisp infrastructure transports and identifies updates, prepares agreed inputs, and tests outcomes; it does not decide the merged meaning.

**Shared execution premise.** Every replica uses the same model and a shared random seed. Pin the entire execution profile too: weights/tokenizer, quantisation, prompt/tool schemas, context construction, runtime/decoder, sampling settings and exact source revision. Derive independent per-merge/per-call random streams from the shared seed and stable operation/context IDs. A single mutable RNG shared across concurrent jobs would make scheduling change results. Include every semantic helper in this scheme. The execution profile is host/driver configuration; the weak interpreter is not asked to choose seeds, sampling parameters or backends inside the merge programme.

**Codebase.** `crdt_lab/merge_history.nl(history, base) -> MergeResult` and `merge_update.nl(state, update, context) -> MergeResult`; helpers `read_intent.nl`, `relate_changes.nl`, `merge_meaning.nl`, `explain_conflicts.nl`, `summarize_history.nl`, and `validate_preservation.nl`. Results contain merged content, source-update references, unresolved ambiguities, and a compact explanation of semantic choices. The latter is recorded output, not hidden reasoning.

**Approach A: history reduction.** Replicate immutable update records containing author, base revision, causal parents and content. Agree on the set and presentation order of updates, then let natlang reduce their meanings against an agreed base. The same full history and profile should yield the same result on supported runtimes. Recompute when new concurrent updates arrive. Test full-context merging against sequential and balanced reduction trees; grouping can change meaning even with reproducible calls. Large-history summaries become agreed, versioned artifacts. Compare reduced history with full-history results before using compaction in live collaboration.

**Approach B: state plus update.** Natlang applies an incoming update to current semantic state, using the author's base state and retained intent evidence when available. This is cheaper and more incremental. Explore pairwise state merging, three-way semantic merge and periodic history reconciliation. Applying A then B may differ from B then A; the shared seed does not remove that path dependence. Record it as an experimental result. A practical synchronisation policy can require replicas to replay a common update order or reconcile against the same history at synchronisation points while the semantic decisions remain entirely natlang-driven.

**Approach C: agreed semantic checkpoints.** Replicas independently generate a proposed merged checkpoint from the same input digest/profile/seed and compare output digests. If they disagree, preserve the alternatives and rerun under the agreed execution environment or explicitly adopt an authorised checkpoint. This is a recovery protocol around semantic merging, not a crisp replacement for merging. Do not silently declare mismatching states synchronised.

**First slice.** Two replicas edit a small task document with overlapping instructions (“move the meeting earlier,” “keep it after lunch”). Compare history-based and incremental variants; show both useful semantic reconciliation and unresolved conflict. Then expand to prose, structured records, code edits and wiki cells.

**Scenarios and acceptance.** Duplicate deliveries, causal reorderings, disjoint edits, delete/edit conflict, contradictory intent, renamed entities, offline branches, model/profile mismatch, context truncation, summary drift and adversarial page text. Measure identical-input reproducibility, post-sync agreement, semantic preservation, conflict quality, latency and replay cost separately. Accept intended ambiguity as unresolved where required. The initial supported runtime must pass repeated seeded replay; cross-backend/browser equality remains a measured capability.

**Important distinction.** Equal inputs plus reproducible execution can produce equal outputs without algebraic convergence under arbitrary histories. Conventional CRDTs make stronger claims based on specific mathematical properties; this project intentionally investigates a different design. [CRDT research overview](https://arxiv.org/abs/1805.06358).

**Training.** Freeze history, presentation order, seed/profile, accepted semantic results and quality rubrics. Do not train on replica agreement alone: replicas can agree on a poor merge. Include contrast pairs where preserving both intentions, choosing a supported interpretation, or reporting unresolved conflict is appropriate.

## P03. Universal build tool with bounded self-repair

**Implementation plan:** [P03 — natlang prerequisites, embedding and delivery](projects/P03_BUILD.md).

**Product.** Build varied projects from semantic goals, maintain correct incremental caches, explain failures, and attempt evidence-based repairs within an explicit policy. “Universal” means extensible toolchain adapters, not immediate compatibility with every existing build system.

**Codebase.** `build_tool/build.nl(goal, workspace, policy) -> BuildReport`; `discover.nl`, `plan.nl`, `diagnose.nl`, `choose_repair.nl`, `verify_repair.nl`. Extend the existing `dependency_plan` pattern. Crisp helpers validate DAGs, identify ready nodes, compute cache keys, reserve output paths and verify patch preconditions. The scheduler executes only the graph that natlang has explicitly selected and that validators accept.

**State.** A `BuildNode` contains declared inputs, toolchain, command or crisp function, environment, prerequisites, outputs and contracts. Build results link artifacts, diagnostics, read dependencies and cache evidence. Separate semantic planning cache from exact action-output cache. Record compiler-discovered dependencies; undeclared observations make an action uncacheable until captured.

**Repair loop.** On failure, natlang inspects bounded diagnostics, proposes a hypothesis and patch or environment change, tests it in a branch, and compares required contracts. Repairs cannot delete a failing test, weaken the acceptance condition, or claim success from exit status alone. Each repair changes source/config hashes and invalidates affected caches. Preserve the failed baseline and cap attempts, tokens, time and patch scope.

**First slice.** Build a tiny mixed source/assets project, reuse cached work after a no-op, invalidate exactly the dependent outputs after a source edit, and repair one generated configuration mismatch. Later add wrappers for existing build tools, remote cache transport and distributed jobs.

**Scenarios and acceptance.** Missing dependencies, cycles, changed compiler, changed environment, interrupted output publication, poisoned cache entry, undeclared file read, flaky test and an unrepairable error. Require correct output digests, causal cache explanations, no false success and a replayable repair history. Compare against a clean rebuild oracle.

**Training and gaps.** Train planning, diagnosis and refusal to weaken requirements separately. This project motivates graph validation, reproducible environments, durable jobs, cache equivalence policies, source patch artifacts and repair-budget semantics.

## P04. Semantic bash terminal

**Implementation plan:** [P04 — natlang prerequisites, embedding and delivery](projects/P04_TERMINAL.md).

**Product.** A persistent terminal where users describe tasks and natlang manages typed plans, tool recipes, working context, diagnostics and results. Media, build and Git routines are libraries available to the session.

**Codebase.** `semantic_terminal/step.nl(acc: Session, item: UserEvent) -> Session`; `interpret.nl`, `choose_recipe.nl`, `plan_actions.nl`, `explain_result.nl`, `recover.nl`. Session state contains workspace grants, current directory, declared environment, jobs, recent artifact references and outstanding intent. It must be serialisable without holding a live shell process as its only source of truth.

**Boundary.** Most execution uses an executable plus argument array, not a generated shell command string. Expose pipelines, redirections and exit conditions as structured recipes with exact compilation. A separate Bash capability supports genuinely shell-dependent scripts under the terminal host's resource policy. Shell parsing and quoting improve feedback; the embedding explicitly chooses direct host execution or isolation. Neither choice is implied merely by using a Bash engine. Never import arbitrary terminal output into the instruction channel.

**First slice.** “Find the failing package, build it, and show a short explanation” in a fixture workspace; then route a media request through P01. Support background jobs, explicit cancellation, typed results and session restart.

**Scenarios and acceptance.** Spaces/metacharacters in filenames, command not found, partial pipeline failure, stale working directory, large output, cancelled job, insufficient scope and a request whose desired action is unclear. Check argv and filesystem effects exactly. Git recipes should inspect status and preserve unrelated changes; higher-impact operations follow the session's declared policy.

**Training and gaps.** Generate paired tasks differing in one precondition, with verified command effects and honest failures. The terminal tests recipe composition, cross-project package imports, durable session context, bounded log retrieval and consistent authorisation without an approval prompt for every routine step.

**Implemented shared path (2026-09-21).** `TerminalNatlangApplication` now
provides the queued reducer/view lifecycle, event multiplexing, durable local
checkpoints, terminal view rendering and model adapter. The semantic terminal
uses it with structured workspace recipes and explicit job/recovery events; the
log investigator and evidence atlas reuse it as streaming and interactive CLI
applications. See `ts-host/TERMINAL_APPLICATIONS.md`.

## P05. Natlang type inference and semantic type checking

**Implementation plan:** [P05 — natlang prerequisites, embedding and delivery](projects/P05_TYPES.md).

**Product.** Infer useful candidate signatures from prose, call sites and surrounding code; inspect whether plausible executions call functions with compatible arguments and use results correctly.

**Codebase.** `type_tools/infer.nl(source, context) -> Suggestion[]` and `check.nl(codebase, scope) -> Diagnostic[]`; `identify_operations.nl`, `infer_bindings.nl`, `trace_paths.nl`, `propose_signature.nl`, `explain_conflict.nl`. Start with explicit frontmatter and existing loader output. Use an indexed source snapshot to retrieve only relevant callees, callers and examples.

**Method.** Natlang proposes a call/binding graph and candidate types with source spans. Exact code checks the proposed obligations against structural `fits` rules and concrete fixtures. Conflicting evidence produces alternatives or an unresolved diagnostic. Observed execution types are evidence, not proof that every path has that type. Record which functions, branches and scenarios were examined.

**First slice.** Suggest missing signatures in copies of existing example codebases, detect deliberately introduced bad argument names and result types, and show evidence for each suggestion. Then handle optional fields, unions, maps/folds, effects and function-copy specialisation. Keep author acceptance separate from runtime enforcement.

**Scenarios and acceptance.** Ambiguous prose, overloaded ordinary words, absent optional inputs, recursive data types, impossible returns, unsound narrowing, hidden effect requirements, edited function behavior and a rare path absent from traces. Check emitted schemas parse, suggested edits load, and known concrete mismatches are found. Evaluate precision and missed defects on held-out project families. Avoid a global “type safe” badge based only on a semantic pass.

**Training and gaps.** Train evidence-to-obligation, counterexample-to-diagnostic and uncertainty handling. Needs stable spans, code search, source revision pinning, an exact checker API, and explicit reporting of analysis limits; it does not require replacing the runtime's type system first.

## P06. Complete natlang IDE

**Implementation plan:** [P06 — natlang prerequisites, embedding and delivery](projects/P06_IDE.md).

**Product.** An IDE whose application behavior, view composition, semantic highlighting, code assistance, run management and training experiment workflows are implemented in natlang. A small browser host renders typed views and emits input events.

**Codebase.** `ide/step.nl(state, event) -> State`, organised into editor, navigator, diagnostics, runner, debugger and dataset/workbench modules. Extend existing `highlighter` and P05. Natlang creates `ViewNode` descriptions with stable IDs; crisp code validates attributes, escapes content, reconciles DOM and binds events. Use cached/incremental highlighting of changed spans so typing does not wait on a complete project interpretation.

**Debugger.** Show function tree, source revision, args/locals/return, pending combinators, line marks, effects and model/tool turns. Step through committed events, jump to origins, compare attempts, and fork into a reset fixture. Distinguish replaying recorded decisions from rerunning the interpreter. Display external effects as historical receipts; moving the UI backward does not undo them.

**Dataset workbench.** Define scenarios, desired behavior, exact checks and semantic rubrics; generate candidate cases with a stronger model; run teacher/student batches; inspect rejects; freeze train/dev/test groups; monitor evaluation and training jobs; activate a candidate checkpoint in a new experiment; compare to the previous checkpoint and restore the selected one. Training engines and GPU processes are host jobs; natlang owns the workflow and interprets reports. Candidate-generated tests need review/admission too.

**First slice.** Browse one codebase, edit a function, display highlighting/type suggestions, run a fixture, inspect its trace, and replay it. Add dataset collection next, then fine-tune management. This bootstraps a useful editor before the editor manages its own training.

**Scenarios and acceptance.** Source changes during a run, invalid frontmatter, stale diagnostic ranges, huge traces, interrupted training, incompatible adapter/model, model reload failure and a worse checkpoint. Verify stable views, source isolation, replay fidelity, dataset lineage and reversible checkpoint selection. Measure interaction latency and interpreter token usage per edit.

**Affordances.** Typed UI schema, incremental work, read-only debugger APIs, durable jobs, stable source spans, checkpoint identity and whole-program trajectory capture. A complete IDE is a later integration milestone, not the first foundation project.

## P07. Natlang package manager

**Implementation plan:** [P07 — natlang prerequisites, embedding and delivery](projects/P07_PACKAGES.md).

**Product.** Discover reusable functions, resolve dependencies, explain compatibility, install reproducibly and propose tested upgrades. Natlang handles intent, package evaluation and upgrade reasoning; exact algorithms handle version constraints, hashes and locked installs.

**Codebase.** `packages/resolve.nl(request, index, policy) -> ResolutionReport`, `install.nl(lock, target) -> InstallReport`, `upgrade.nl(workspace, constraints) -> UpgradePlan`. Package metadata includes source exports, named types, required effects, runtime/schema compatibility, source licenses, test fixtures and immutable content hashes.

**Design.** Bootstrap with a local package index and directory/tar artifacts. The solver accepts candidate constraints and returns a graph or an unsatisfiable explanation. A lockfile pins transitive packages, crisp evaluator dependencies and toolchain requirements. Install materialises a directory layout usable by current lexical `uses` links. Installing a package does not grant its requested capabilities or run arbitrary post-install code. Explicit build steps use the package host's declared execution adapter; a job service is optional.

**First slice.** Extract two existing helper libraries, resolve and install them offline, then reproduce the same dependency tree from the lockfile. Add an intentionally incompatible upgrade and demonstrate useful diagnostics. A public registry, publishing protocol and signatures can follow local reproducibility.

**Scenarios and acceptance.** Conflicting versions, missing exports, transitive effects wider than the app grant, dependency cycles, archive traversal, digest mismatch, registry unavailability and interrupted installation. Require atomic installation, unchanged prior lock on failure, no ambient imports and a repeatable source tree.

**Training and gaps.** Train semantic discovery and migration advice, with exact solving and tests as the oracle. The package manager supplies infrastructure for every later project but should not block the first applications, which can use local links.

## P08. Games: economic simulation, actor combat and general NPCs

**Implementation plan:** [P08 — natlang prerequisites, embedding and delivery](projects/P08_GAMES.md).

These are three distinct vertical slices sharing an event-driven world host, seeded environment randomness, typed actions, exact world transitions and replayable sessions. Natlang runs each actor's policy and coordinates higher-level behavior.

**A. Economic simulation.** `economy/tick.nl(world, observations) -> IntentBatch` invokes agents that interpret scarcity, propose trades, negotiate and adapt production. Crisp code clears trades under explicit rules, tracks integer money/quantities and enforces conservation. Begin with three merchants and two goods; expand to supply chains, contracts and shocks. Test bankruptcy, unavailable goods, contradictory offers and dishonest messages. Measure conservation exactly and strategic quality empirically; do not infer good economics from fluent dialogue.

**B. Actor-based fighting game.** `arena/decide.nl(perception, memory) -> CombatIntent` chooses goals, tactics and short action scripts. Crisp code handles physics, collision, legal actions, cooldowns and frame timing. Start turn-based, then run semantic decisions at a lower frequency than the render/physics loop. Slow decisions use an explicit safe fallback rather than stalling the game. Test simultaneous moves, stale observations, cancelled attacks and resource exhaustion. Record decision deadlines, action legality and fairness under identical observations.

**C. General NPCs.** `npcs/react.nl(event, memory, goals) -> ResponsePlan` decides dialogue, commitments and world actions using permission-scoped memory retrieval. Exact code validates inventory, access, quest state and action preconditions. Begin with one persistent shopkeeper extending the current example; then give a small village multiple interacting NPCs. Test conflicting memories, failed promises, missing evidence, malicious player text and interruptions. Long-term consistency is a scored property, not guaranteed by a vector database.

**Shared architecture.** Actors own private state and serial mailboxes; actions become messages to an authoritative world step. Run independent policies concurrently with isolated random streams, then resolve simultaneous actions by an explicit world rule. Never allow an NPC to mutate another actor's private state through a retrieved handle.

**Training and gaps.** Freeze world seeds, observations, legal actions and outcomes. Include rejected moves and sensible fallbacks. Evaluate unseen worlds and goals, not just new wording for the same scene. These prototypes test scheduling, delayed observations, state retrieval, pause/resume and budgets under interactive deadlines. Multiplayer transport and high actor counts follow successful small simulations.

## P09. Spell programming game with a specialised fine-tune

**Implementation plan:** [P09 — natlang prerequisites, embedding and delivery](projects/P09_SPELLS.md).

**Product.** Players invent spells in natural language; a natlang interpreter translates them into meaningful combinations of game actions. The game should reward expressive composition while preserving exact resource and world rules.

**Codebase.** `spells/cast.nl(utterance, world_view, caster, rules) -> SpellResult`; `interpret_spell.nl`, `compose_effects.nl`, `resolve_ambiguity.nl`, `describe_outcome.nl`. A typed `SpellPlan` expresses allowed targets, effects, magnitude, duration, ordering and resource cost. Crisp validation rejects unavailable targets, impossible combinations or excessive cost before any world effect occurs.

**Execution.** Natlang decides how “a ring of frost that slows pursuers but spares allies” maps to allowed mechanics. The host validates and atomically applies the accepted plan, or reports why it cannot be cast. A late world-state change invalidates or revalidates the plan against its snapshot. Animation/audio consume approved events and artifacts.

**First slice.** A small arena with damage, movement, shielding and elemental effects; show a cast preview and explanation. Then introduce persistent effects and creative combinations. Avoid making the first student learn unrestricted new mechanics invented during play.

**Training.** Teacher generates diverse utterances and candidate plans inside a frozen rule set. Exact simulator checks legality and cost; separate semantic checks score correspondence to requested intent. Include impossible spells, ambiguous references, negations, metaphor, paraphrases and attempts to escape the game rules. Split by compositional template and mechanic combinations. Fine-tune a spell interpreter variant and compare with the general student on the same unseen scenarios.

**Acceptance and gaps.** Require legal execution, no unvalidated world changes, bounded latency and graceful abstention. Score creativity and intent fulfilment separately. This project needs robust enum/record outputs, authoritative simulation, branchable scenarios and domain-specific training without contaminating general interpreter evaluation.

## P10. Natlang notebook with data affordances

**Implementation plan:** [P10 — natlang prerequisites, embedding and delivery](projects/P10_NOTEBOOK.md).

**Product.** A notebook with natlang cells, exact snippets, SQL queries, artifact previews and inspectable data lineage. Natlang orchestrates data cleaning, analysis and explanation; crisp code computes exact results.

**Codebase.** `notebook/step.nl(state, event) -> NotebookState`, `run_cell.nl`, `infer_dependencies.nl`, `interpret_table.nl`, `choose_transform.nl`, `explain_result.nl`. A cell has stable identity, source revision, declared inputs, outputs, execution profile and dependency edges. Natlang can propose dependencies from prose; the runner validates explicit bindings before execution.

**State.** Tables live as SQLite snapshots or artifacts, with typed summaries in the object tree. Cells reference versions, not mutable global variable names with hidden execution history. Editing a cell invalidates descendants; rerunning creates new output revisions. Effectful cells have explicit replay policies and receipts. SQL is parameterised and scoped to the cell's granted database/snapshot.

**First slice.** Import two messy CSV fixtures, use natlang to map labels and explain ambiguities, join/aggregate in SQL, and emit a reproducible report. Then add interactive views, reusable cell libraries, long jobs and branching analysis.

**Scenarios and acceptance.** Cell cycles, stale outputs, changed input files, missing columns, ambiguous units, huge query results, expensive queries, cancelled execution and side-effectful reruns. Require accurate data lineage, exact numeric outputs, visible uncertainty and deterministic invalidation. Export a notebook bundle that reproduces results from its manifest.

**Training and gaps.** Teach semantic data operations and selection of exact computation with held-out schemas. Needs explicit cell loading and bindings, data access through selected engines, dependency planning and typed views. Native tables may remain in eval; durable job progress is a later host facility. A notebook cell graph is explicit orchestration data; it is not a parser that executes natlang bodies behind the model's back.

## P11. Collaborative executable wiki and natlang web server

**Implementation plan:** [P11 — natlang prerequisites, embedding and delivery](projects/P11_WIKI.md).

**Product.** Markdown-like pages contain editable prose, assets and natlang cells. The frontend runs page programs, renders their views and synchronises collaborative changes using P02's semantic merging. Server workflows are also natlang, extending the existing webserver.

**Codebase.** `wiki/handle.nl(state, event) -> State`, `render_page.nl`, `run_cell.nl`, `merge_page.nl`, `review_revision.nl`, `resolve_links.nl`. Pages store stable block/cell IDs, source revisions, dependency bindings, execution profile, grants and output artifacts. Natlang interprets page behavior and merges user intent; crisp code parses document boundaries, transports updates, validates revisions and renders approved view data.

**Replication.** Use the agreed P02 history-based or state/update policy, shared seed and pinned model profile. Concurrent prose edits, code edits and cell moves need distinct semantic evidence. A merged code block must load and pass relevant checks before it replaces a runnable version. Preserve conflicted alternatives and their authorship. Pin a page revision for each execution so a live edit cannot change instructions halfway through a run.

**Client execution.** Treat browser inference as a measured porting project. Verify model/operator support, tokenizer, tool decoding, memory, startup, context size, responsiveness and result parity on actual target devices. WebGPU is an available compute API, not proof that this particular interpreter stack will run acceptably or produce bit-identical results. [WebGPU specification](https://www.w3.org/TR/webgpu/). Use a common remote execution profile or agreed checkpoint recovery when browser replicas cannot reproduce merge decisions; local page execution can still remain useful.

**Authority.** Page-authored programs receive explicit grants. Public content does not inherit the viewer's private state, filesystem or network privileges. Render user HTML inertly unless separately allowed. Authentication, transport and resource checks remain host responsibilities; routing and application policy live in natlang.

**First slice.** Two clients edit one page containing prose and one pure natlang cell. Demonstrate offline edits, semantic reconciliation, pinned execution and output provenance. Add server effects, assets, larger pages and client inference afterward.

**Acceptance/training.** Test disconnect/reconnect, profile mismatch, semantic merge disagreements, invalid merged code, stale results and malicious page instructions. Require no lost source updates, visible unresolved conflicts, reproducible agreed-profile merges, bounded cell execution and permission isolation. Train merge quality, page behavior and runtime execution as separate linked tasks.

## P12. Server log anomaly tracker and investigator

**Implementation plan:** [P12 — natlang prerequisites, embedding and delivery](projects/P12_LOGS.md).

**Product.** Consume logs and metrics, identify unusual patterns, investigate evidence, summarise incidents and escalate according to policy. Natlang decides which anomalies matter and how to investigate; exact code performs parsing, counting, windowing and delivery deduplication.

**Codebase.** `log_watch/step.nl(state, event) -> State`, `assess_window.nl`, `form_hypotheses.nl`, `request_evidence.nl`, `correlate.nl`, `write_incident.nl`, `choose_escalation.nl`. Incident records distinguish observations, hypotheses, disconfirming evidence, actions and unresolved questions.

**Pipeline.** Host adapters normalise structured events and preserve raw log artifacts. Crisp windows compute rates and baselines. Natlang reads compact summaries, searches relevant raw spans, compares deployments/config changes and chooses whether to open or update an incident. A stronger model can inspect an unusually complex case through a bounded help request. External notification is a declared policy-controlled effect with an idempotency key.

**First slice.** Replay fixture service logs containing a deployment regression, one harmless burst and a missing-data interval. Emit a local incident artifact first; add configured notification sinks later.

**Scenarios and acceptance.** Clock skew, out-of-order/duplicate logs, delayed metrics, partial outage, flood/backpressure, log text impersonating instructions, secrets in logs and repeated escalation after restart. Check window/cursor correctness and notification multiplicity exactly; score detection precision/recall and causal claims against seeded incidents. Unobserved root causes remain hypotheses.

**Training and gaps.** Include benign anomalies and evidence that disproves the initial diagnosis, not only clear incidents. Uses stream Fold and crisp search over bounded observations. A production deployment adds persistent cursors, event-time policy and acknowledgement-aware escalation; model help is an optional environment binding. Retention and indexing must accommodate sustained volume without stuffing logs into prompts.

## P13. Data reconciliation and migration studio — new

**Implementation plan:** [P13 — natlang prerequisites, embedding and delivery](projects/P13_DATA.md).

**Product.** Reconcile inconsistent records across datasets, suggest schema mappings and execute explainable migrations. This extends the existing reconciliation example into a practical application.

**Codebase.** `data_studio/reconcile.nl(sources, target_schema, policy) -> MigrationPlan`; `map_fields.nl`, `match_entities.nl`, `resolve_conflicts.nl`, `explain_change.nl`. Natlang judges whether “client,” “account holder” and “customer” refer to corresponding concepts, and whether two records plausibly identify one entity. Exact code computes candidate joins, validates types, applies transactions and maintains source-to-target lineage.

**Execution.** Generate a proposed patch set with evidence and uncertainty. Preview changes on a snapshot, run constraints, and publish only accepted patches through a transactional adapter. Ambiguous matches stay unresolved; semantic similarity alone cannot merge identities. Every output field traces back to source rows or a recorded derivation.

**First slice.** Two fixture customer/order exports with different field names, duplicate entities and incompatible units. Produce a checked target database plus a review queue. Later add schema migrations, repeat sync and incremental reconciliation.

**Scenarios and acceptance.** Missing IDs, same-name different people, conflicting timestamps, unknown units, null versus empty, duplicate imports, mid-transaction failure and a changed source after preview. Require accounting/row constraints, idempotent re-import, no silent dropped rows and complete lineage. Measure false merges separately from missed matches.

**Training and gaps.** Build counterfactual record pairs and held-out schemas. This exercises SQL transactions, semantic matching, exact patch application, evidence handles and uncertainty presentation; it is a strong early notebook workload.

## P14. Evidence notebook and research atlas — new

**Implementation plan:** [P14 — natlang prerequisites, embedding and delivery](projects/P14_EVIDENCE.md).

**Product.** A searchable local collection of documents, code, runs and observations that answers questions with inspectable evidence and maintains competing claims over time.

**Codebase.** `evidence_atlas/answer.nl(question, collection, policy) -> EvidenceAnswer`; `plan_search.nl`, `extract_claims.nl`, `compare_sources.nl`, `resolve_question.nl`, `identify_gaps.nl`. Natlang plans retrieval, interprets relevant passages and explains disagreements. Exact adapters index documents, retrieve spans, verify source hashes and enforce collection permissions.

**State.** Store claims separately from source facts, with provenance, temporal scope, support/contradiction links and review status. A summary cannot overwrite the original. Begin with text/full-text search; add vector retrieval only after evaluating recall on a fixed question set.

**First slice.** Index selected natlang specs and run reports, then answer “Which failures motivated this harness change?” with links to exact source evidence. Expand to external documents and multimedia-derived observations later.

**Scenarios and acceptance.** Conflicting versions, outdated claims, inaccessible evidence, deleted sources, near-duplicate documents, misleading retrieval and a question with no supported answer. Require valid citations and permission filtering; score answer support and retrieval recall independently. Explicit unknowns are correct outcomes where evidence is absent.

**Training and gaps.** Freeze the corpus snapshot and question family; hold out entire source collections to limit leakage. Teach search refinement, conflict preservation and justified abstention. This directly exercises searching the program's own state and history without violating function privacy.

## P15. Stateful API workflow composer — new

**Implementation plan:** [P15 — natlang prerequisites, embedding and delivery](projects/P15_WORKFLOWS.md).

**Product.** Compose typed service operations from a user goal, discover how to recover from failures and explain partial completion. The motivating applications include resource provisioning, inventory synchronisation and document-processing workflows.

**Codebase.** `api_workflows/run.nl(goal, services, policy) -> WorkflowReport`; `choose_operations.nl`, `bind_inputs.nl`, `interpret_response.nl`, `plan_compensation.nl`, `reconcile.nl`. Service schemas and recorded examples are explicit inputs. Crisp adapters validate requests/responses, implement transport, manage credentials and maintain receipts.

**Execution.** Natlang selects a bounded workflow and recovery branches. Persist each action intent before dispatch. Use per-operation idempotency where available, reconcile unknown results and compensate with new operations when policy permits. Extend the existing `order_saga` patterns into durable infrastructure. Credentials stay in the host and are represented to the interpreter by resource grants.

**First slice.** A fake three-service workflow that reserves inventory, processes a fixture payment and schedules shipping. Lose an acknowledgement, restart, and demonstrate recovery without a duplicate charge. Real service adapters follow that fault-complete simulation.

**Scenarios and acceptance.** Timeouts before and after commit, expired authentication, incompatible schema, rate limits, partial failure, compensation failure and unsupported cancellation. Assert exact delivered effects, reported uncertainty and preservation of unresolved obligations.

**Training and gaps.** Teach precondition checks, binding, semantic response interpretation and recovery choice. This is the strongest test of effect receipts, outboxes, remote uncertainty and durable resumability; do not use successful HTTP status alone as an end-to-end oracle.

## P16. Simulation and experiment laboratory — new

**Implementation plan:** [P16 — natlang prerequisites, embedding and delivery](projects/P16_EXPERIMENTS.md).

**Product.** Define and compare policies in simulated worlds, explore counterfactuals and explain which observations distinguish competing hypotheses. Useful for games, scheduling, operations and interpreting natlang behavior itself.

**Codebase.** `experiment_lab/compare.nl(question, world, candidates, budget) -> ExperimentReport`; `design_experiment.nl`, `choose_scenarios.nl`, `analyse_results.nl`, `propose_followup.nl`. Natlang selects meaningful comparisons and interprets outcomes. Crisp code implements world transitions, controlled RNG, metrics and statistical calculations.

**Execution.** Freeze experiment manifests and use matched environmental seeds across policies. Give environment, actor and semantic-decision randomness separate named streams. Branch state at explicit checkpoints. Record the complete candidate policy revision and execution profile, not just a human-readable name.

**First slice.** Compare two economic or scheduling policies across a small predeclared scenario suite, including an intentionally adverse case. Then add adaptive experiment selection without modifying the held-out evaluation set.

**Scenarios and acceptance.** Seed collision, incomparable initial conditions, missing runs, early termination, policy changes mid-batch and conclusions unsupported by sample size. Require reproducible metrics and clear limits on conclusions; preserve failed runs in denominators.

**Training and gaps.** Train experiment design and evidence-based interpretation, with exact computations verified separately. This project exercises fork/replay, isolated RNG, model budgets and evaluation discipline. It also supplies infrastructure for comparing P02 merge strategies without confusing agreement with semantic quality.

## P17. Living specification and test explorer — new

**Implementation plan:** [P17 — natlang prerequisites, embedding and delivery](projects/P17_TESTS.md).

**Product.** Read a natlang codebase and its stated behavior, generate challenging scenarios, explore execution paths, minimise failures and propose specification clarifications.

**Codebase.** `spec_explorer/explore.nl(program, contracts, budget) -> Findings`; `identify_obligations.nl`, `invent_scenario.nl`, `assess_coverage.nl`, `explain_counterexample.nl`. Natlang hypothesises missing cases and interprets failures. Crisp code builds typed fixtures, executes them in a reset world, checks invariants and shrinks inputs while retaining the observed failure.

**First slice.** Audit the current dependency planner and saga against cycle, duplicate, ordering and failure contracts. Produce a minimal reproducible case and an evidence-linked finding rather than automatically changing the program.

**Scenarios and acceptance.** Incorrect test oracle, impossible generated inputs, flaky model outcomes, a changed source during minimisation and an invalid “fix” that weakens the contract. Freeze seed/profile when reproducing semantic failures and report reproducibility rates. A model-written test is a proposal until its expected behavior is independently justified.

**Training and gaps.** Use accepted counterexamples to create contrast families: correct execution, wrong shortcut, blocker and valid recovery. Keep the entire family in one dataset split. This application is especially valuable for the planned foundation/app feedback loop and gap register.

**Longer scope.** Add behavioural compatibility checks between package versions, systematic schedule exploration and regression selection by affected contracts. It needs scenario manifests, fault injection, precise source hashes, trace inspection and a crisp invariant library; it must not silently become the same teacher both inventing and grading its own answers.

## P18. Document compiler and publishing system — new

**Implementation plan:** [P18 — natlang prerequisites, embedding and delivery](projects/P18_PUBLISHER.md).

**Product.** Compile semantic source material into consistent reports, manuals, tutorials or interactive publications. Natlang decides structure, emphasis and cross-references; crisp renderers enforce layout schemas, escaping and asset linking.

**Codebase.** `publisher/publish.nl(source, audience, format, rules) -> Publication`; `outline.nl`, `compose_section.nl`, `check_consistency.nl`, `adapt_view.nl`. An intermediate document tree separates claims, citations, tables, figures, code cells and navigation. Rendering to HTML and a printable representation should preserve this shared content identity.

**First slice.** Turn frozen natlang scenario results into an experiment report with exact tables, source citations and a readable narrative. Produce HTML and Markdown from one checked tree; add print/PDF and interactive cells later.

**Scenarios and acceptance.** Missing asset, broken link, unsupported assertion, conflicting numbers, long tables, code escaping and source changes after drafting. Exact checks verify references, totals, schema and asset existence; visual inspection and semantic review assess readability. Publishing is a separate effect from preparing an artifact.

**Training and gaps.** Teach source-grounded composition, coherent decomposition and adaptation without inventing evidence. Capture intermediate outlines and revisions only if independently checked. This exercises typed UI/document trees, binary assets, source provenance and renderer consistency, and can publish the outputs of most other projects.

## P19. Personal workflow and scheduling engine — new

**Implementation plan:** [P19 — natlang prerequisites, embedding and delivery](projects/P19_SCHEDULING.md).

**Product.** Manage tasks with soft preferences, dependencies and real-world disruptions. Natlang interprets priorities and explains tradeoffs; exact scheduling checks enforce availability, duration, dependency and resource constraints.

**Codebase.** `workflow_planner/step.nl(state, event) -> State`; `interpret_request.nl`, `rank_tradeoffs.nl`, `propose_schedule.nl`, `explain_replan.nl`. Keep preferences distinct from hard constraints. Model calendar/time-zone observations as versioned inputs, not guessed facts.

**First slice.** Schedule a fixture workday, receive an interruption and replan while preserving required commitments. Use a local simulated calendar; integrate real calendar/task services only after receipt and reconciliation semantics are exercised.

**Scenarios and acceptance.** Infeasible requests, contradictory priorities, missing durations, daylight-saving changes, duplicated events, rescheduled dependencies and a meeting changed after preview. Check hard constraints exactly and score preference satisfaction separately. An infeasible request produces alternatives or a blocker rather than fictitious free time.

**Training and gaps.** Include paired feasible/infeasible cases and explanations grounded in the exact conflict set. The project tests logical clocks, durable timers, constraint interfaces, state updates and appropriate handling of missing information. Human decisions arrive as typed events, allowing workflows to suspend without holding an inference slot.

## P20. Repository maintenance and migration agent — new

**Implementation plan:** [P20 — natlang prerequisites, embedding and delivery](projects/P20_REPOSITORIES.md).

**Product.** Perform bounded codebase migrations, dependency updates and maintenance tasks while preserving behavior and unrelated changes. This combines semantic source understanding with the build tool, package manager and test explorer.

**Codebase.** `repo_maintainer/migrate.nl(request, snapshot, policy) -> ChangeReport`; `find_uses.nl`, `plan_change.nl`, `edit_source.nl`, `interpret_tests.nl`, `review_diff.nl`. Natlang plans and writes changes. Crisp adapters create isolated workspaces, apply patches with source-hash preconditions, run format/build/tests and collect diffs.

**First slice.** Rename or change the signature of a helper across one natlang example and all its callers, preserving scenario outcomes. Then migrate a library version that changes one capability or type contract.

**Scenarios and acceptance.** Dirty workspace, stale patch, generated files, hidden callers, a passing but irrelevant test suite, ambiguous migration instructions and a repair loop that changes requirements. Require source isolation, intended diff scope, correct updated calls and unchanged protected contracts. Report validation coverage rather than treating a green command as proof of all behavior.

**Training and gaps.** Freeze before/after revisions and test evidence. Hold out repository/task families. Include failures that must remain blocked and cases where no change is necessary. This project tests filesystem grants, source search, package compatibility, patch provenance, bounded repair and explicit publication boundaries. It should mature after the build tool and IDE source-edit APIs are usable.

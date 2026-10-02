# A natlang program that improves natlang programs

Status: complete-scope implementation assignment, revised 2026-09-30. The APIs and commands marked proposed below are not implemented. Implement this entire plan in one continuous effort, including the API refactor and migration of current behavior.

## 1. Outcome and interpretation

Execution mandate: **implement the whole plan completely in one go**. There are no intermediate releases, approval gates, phase sign-offs, pilot-only stopping points, or decisions to defer required work. Every work item, T1–T6 capability, training-data migration/generation task, both student learning rounds, documentation update and verification is part of this assignment. Continue automatically through failures by fixing, rewriting, replaying or recollecting while progressing other available work. Internal dependencies determine execution order; they do not create a place to stop and ask whether to proceed.

Runtime candidate acceptance, independent scoring, source/type checks, dataset admission, finite budgets and atomic transactions remain product behavior. They are not implementation approval checks. Evaluation levels and adopting frozen improver versions between runtime runs describe the language's execution semantics, not phased delivery. Report empirical results honestly and pursue the stated improvement objectives within declared experiment allocations; measurements never suspend implementation of the remaining scope. Report actual external blockers explicitly while completing all independent work.

The primary goal is an excellent **natlang-native test case**: a real semantic program that edits, evaluates and returns programs, exposes weaknesses in our language/runtime/tooling, and drives refinements that make this workflow natural for underpowered models. Ship a semantic programming system in which small models author and transform executable programs and improve the authored improver through the same bounded evaluation machinery. Measure target and failure-derived student-learning benefits honestly; those outcomes are evidence about the system, not the organizing principle or a reason to hide semantic work behind host code.

Prefer additional language/runtime/compiler/tool/data work that makes the native program cleaner over preserving an awkward API or maximizing a benchmark by moving the algorithm into TypeScript host services. Treat repeated workaround code, confusing return/effect semantics, excessive context plumbing and model-facing bookkeeping as refactor opportunities: simplify the shared system, rewrite the improver to use it, and migrate/generate its training examples in the same assignment. Inspect complete native traces and authored source to find these problems; do not add a separate evaluation framework merely to score architectural cleanliness. The resulting improver must remain ordinary authored natlang, including its experiment planning and source transformation. Keep exact arithmetic, independent scoring and native execution authority in exact code. A disappointing model score should first prompt inspection of the native expression, contracts, diagnostics, context and training coverage; it does not justify replacing the native program with a successful host-side optimizer.

The improvement application is an authored, inspectable natlang program. It takes a target program, an evaluation suite and an improvement policy; produces validated candidate programs; evaluates them through the actual runtime; selects a result; and emits runnable selected source, its source patch, an artifact and evidence. Developers inspect, edit, test and adapt the improver using the same tools they use for their own natlang programs. No-change results carry the baseline source and an empty patch; adoption remains an explicit operation.

The broader deliverable is **program transformation as an ordinary semantic programming task**: a program is inspectable source that a directory reducer can edit, evaluate through a capability supplied from above, and return as new source. The optimizer is the first substantial application of this facility. API design must prioritize very underpowered coding models: few concepts, concrete argument names, short examples, explicit effects, and useful local errors. Refactor existing APIs when that reduces mistakes; preserving the present API is not a design constraint.

Every API change includes a training-data deliverable: directly migrate applicable existing examples and trajectories, research additional task sources, and generate newly executed examples for the changed behavior. This applies to public methods, model tools, declarations, prompts, return/error shapes, effect semantics, compiler diagnostics and later extension APIs. Data migration and new coverage are part of completion, not documentation follow-up; section 3.6 defines the common check.

“Entirely a natlang program” means the application consists of `.nl` functions and ordinary TypeScript orchestration/helpers compiled by natlang. Exact arithmetic, hashes, selection, file operations and transactions belong in TypeScript. Diagnosis, credit assignment, instruction rewriting and discretionary planning belong in natural-language functions. The compiler, interpreter, model transports and durable storage remain runtime infrastructure, just as they do for any natlang application.

The application must own the improvement algorithm. A `.nl` wrapper around the existing `optimize()` service does not satisfy this plan. Conversely, asking a model to reproduce Pareto arithmetic, maintain request counters or decide whether a hash matches would make the port less reliable without making it more idiomatic.

Implement all of the following together as one complete deliverable:

0. **Edit/evaluate/return integration examples:** executable folder-based `iterateOn` examples edit targets, evaluate accepted/rejected proposals and return selected source. Include instruction changes and helper extraction as continuously exercised integration examples, not a separate prototype delivery.

1. **Instruction and guidance improvement:** a complete replacement implementation of the current lifecycle, preserving frozen signatures, captures, slots, helper topology and capabilities.
2. **Structural program improvement:** an explicit mode for editing authored TypeScript, helper decomposition and callable contracts in private candidate snapshots. Preserve the declared external behavior and verify changes independently. Structural changes produce a new build and a source-change artifact; they cannot masquerade as a compatible instruction overlay.
3. **Adaptive semantic development:** authored experiment planning, semantic conflict resolution, transformation contracts and reusable reducers; working implement/repair/simplify, counterexample-guided and portfolio applications.
4. **Frozen-copy self-improvement and demonstrated learning:** transform frozen improver copies, adopt them between runs, and complete two failure-driven student training/evaluation rounds with independent evidence of learning benefit.

Instruction and structural modes, adaptive improvement, all T1–T6 packages, frozen-copy optimizer self-improvement, the authored default engine, and measured target/student-learning experiments are one indivisible scope. Complete them in the same implementation effort; do not deliver an instruction-only release or stop after integration examples.

## 2. What exists and what must change

The [adaptation implementation plan](ADAPTATION_SYSTEM_IMPLEMENTATION.md), [author guide](../docs/ADAPTATION.md) and [implementation evidence](../docs/ADAPTATION_IMPLEMENTATION_EVIDENCE.md) are the starting contract. Preserve its deployment, compatibility, evaluation and accounting guarantees.

| Current implementation | Port treatment |
| --- | --- |
| `optimization/optimize.ts` owns search, selection, caching and finalization | Move the application state machine and strategy orchestration into the authored improver. Keep only reusable infrastructure behind host services. |
| `optimization/proposer.ts` directly calls a reflection driver and parses text/JSON strings | Replace with typed `.nl` diagnosis and proposal functions, retaining exact validation in TypeScript. No JSON-encoded static segment arrays inside strings. |
| `strategies/gepa.ts`, component selector, dependency and Pareto helpers | Reuse or move the exact algorithms into the improver's authored TypeScript helpers with existing attribution and deterministic tests. |
| `evaluation/*` already executes actual targets in fresh workers | Expose split-restricted, budget-accounted evaluation capabilities with opaque evidence references. |
| Artifact schemas, compiler descriptors, `ProgramView` and export/revalidation | Reuse for instruction mode; extend candidate compilation and provenance for structural mode. |
| RunStore, caches, ledgers and checkpoints | Extract typed durable operation primitives without hiding the search algorithm in a service. |
| Runtime services, callable-folder scoping, traces and `iterateOn` | Reuse for the improver; add operation journaling, role routing and evidence pagination where needed. |

Important present limitations: the runtime normally selects one model configuration per task; request roles are `executor`, `reflection` and `judge`; `iterateOn` retains an in-memory trajectory and can invoke a model progress judge; native eval is trusted code rather than a sandbox; current instruction artifacts and revalidation reject changed contracts. Design against these facts rather than assuming missing behavior already exists.

## 3. Architecture

There are two separate execution contexts:

- **Improver context:** the frozen optimizer program, its model configuration, task frame, scoped services, search state and optional independently selected optimizer artifact.
- **Target context:** the candidate program, its deployment executor, fresh case state, candidate artifact/snapshot and target service declarations.

Target evaluation starts in fresh workers. It is not a nested target call in the improver's frame. Target instructions, captures, runtime edits and guidance cannot accidentally become the improver's effective program. Correlation IDs connect the traces; authority and mutable state remain separate.

Proposed layout:

```text
applications/program-improver/
  main.ts                         # supply capabilities and invoke authored folder iteration
  types.ts                        # finite typed requests, proposals and search state
  natlang.d/
    improve-step.nl               # directory reducer: evidence -> experiment -> next state
    rewrite-program.nl            # directory reducer: edit target source, return explanation
    diagnose.nl                   # failure evidence -> hypotheses with citations
    propose.nl                    # hypotheses + contracts -> instruction changes
    plan-experiment.nl            # required bounded adaptive experiment planning
    review-conflict.nl            # required semantic conflict-resolution proposals
    explain-result.nl             # evidence-backed human explanation
    select.ts                     # exact seeded strategy selection
    advance.ts                    # exact phase transitions and acceptance
    candidates.ts                 # exact proposal assembly and conflict handling
    statistics.ts                 # exact paired metrics and uncertainty summaries
    evidence.ts                   # context assembly and paged evidence access
    reporting.ts                  # factual report construction
  suites/                         # improver behavior and meta-evaluation fixtures
ts-host/src/improvement/
  host.ts                         # compiler/evaluator/storage adapters
  services.ts                     # scoped typed service declarations
  operations.ts                   # durable operation journal and cancellation
  routing.ts                      # model roles and shared usage accounting
```

The layout is proposed. Compile the application normally; package authored source, generated declarations, service contracts and notices. The authored program expresses its search loop with `folder.iterateOn` and the explicit proposal/evaluation/acceptance workflow below. Exact helpers enforce arithmetic and mandatory checks; they do not hide the search procedure in a native optimize service. `main.ts` supplies capabilities and invokes the program. Callable-folder TypeScript follows the finite-iteration and recursion policies; the restricted generated-program profile also covers editable application modules. Semantic functions never recursively evaluate themselves in the same execution context.

### 3.1 Service boundary

Expose primitives that require native authority, not a generic “improve everything” endpoint:

| Capability | Operations and restrictions |
| --- | --- |
| Program repository | Open a private target folder; inspect inventory and source; freeze drafts, register immutable proposals, and compile snapshots with diagnostics. |
| Evidence reader | Page approved training outcomes, observations and invocation traces by stable reference; bounded summaries reference their full underlying records. |
| Evaluator | Evaluate a registered, validated candidate on permitted case IDs/seeds in fresh workers; share the parent run's accounting and cancellation. |
| Run journal | Begin/resume operations, store immutable typed values, append events and atomically commit a checked next state against an expected revision. |
| Budget/clock | Read actual remaining allowances and reserve/release authorized operation capacity; mutations are host validated. |
| Finalizer | Freeze a selection, issue a restricted holdout evaluation ticket, validate its evidence and produce the final artifact/report/export. |

Use opaque run/candidate/evidence IDs resolved by the host. Models cannot supply an arbitrary filename and have it treated as an evidence record or executable project. IDs are references, not serialized worker handles or closures.

Scoping separates diagnosis/proposal services from finalization. Diagnosis receives training evidence; planning may see approved aggregate validation summaries, but no held-out case text, expected values or raw test traces. The improver's callable folder contains its own code only. Target source is a separately authorized editable folder, with a read-only view for diagnosis; expected answers and suite implementation are outside that folder. Finalization is an exact application step after selection is frozen, without another opportunity for semantic rewriting.

These restrictions organize the application's exposed authority. Existing trusted native eval and Node workers do not provide hostile-code containment. If accepting untrusted target repositories becomes a supported use case, add a process/container execution boundary and restricted filesystem/network/model credentials as a separate host feature; service scopes alone are insufficient.

### 3.2 API alternatives and small-model selection check

Commit to `folder.iterateOn` with explicit `propose`/`evaluate`/`accept` as the initial architecture. The alternatives below are targeted remedies for observed small-model failures, not a requirement to build five competing APIs before implementation:

| Option | Model-facing shape | Benefit | Cost / experiment |
| --- | --- | --- | --- |
| Folder-owned iteration, recommended starting design | `folder.iterateOn(step, initialState, ...args).until(done)` | Files and typed state advance together; the model edits the familiar `folder` and returns only next state. | Requires folder-aware checkpointing and compiler inference. Test whether models correctly distinguish proposal, acceptance and iteration completion. |
| Programs returned as values | `transform.iterateOn(program, policy).until(done)`; every transform returns a new source-bearing program value | Uniform value semantics, easy branching and composition. | Weak models must construct or return program values after editing. Compare a compiler-injected `program` binding against explicit snapshot construction. |
| Explicit proposal workflow | `folder.propose(rewrite, ...args)`; `evaluator.evaluate(proposal.folder, request)`; `folder.accept(proposal)` | Three concrete verbs expose the actual decisions and prevent rejected edits from leaking. Composes with folder iteration. | More calls than an integrated helper. Measure stale-proposal and forgotten-accept errors. |
| Integrated experiment helper | `folder.experiment(rewrite, evaluator, request, ...args)` returns proposed source plus evidence | Removes snapshot/build plumbing from model-authored code. | Keep acceptance in authored code; the helper must not hide strategy, select a parent, or promote a candidate. Compare against the explicit three-verb workflow before adding another public method. |
| Constrained semantic workflow builder | An authored list of diagnose/edit/check/evaluate/decide steps with typed edges | Minimal coding demand and stronger static validation. | Adds a second representation and constrains adaptive decomposition. Prototype only if ordinary short TS/NL programs remain unreliable. |

Implement the committed surface immediately and probe accept, reject, no-change, compile repair, stale base, zero-step completion, cancellation, and returning selected rather than last-attempted source as part of ongoing integration. Investicheck at most two targeted alternatives where observed failures suggest a remedy; this investigation must not hold up other work. Publish first-attempt correctness, correction requests, spend, completion and source/evidence integrity using matched tasks and allowances. Keep one concrete public contract and revise it directly when evidence supports a change, updating implementation and training data in the same effort. There is no waiting period or API-selection sign-off.

Use the private-branch return semantics specified below. Probe whether models understand them; compare receiver mutation only if observed failures implicate that distinction. Implement one behavior, one return shape and one documented spelling consistently across the system. Do not defer the implementation merely because another behavior remains conceivable.

Keep host bookkeeping out of the ordinary model task. Generate source/build IDs, hashes, operation IDs, split tickets and journal revisions in exact code; models receive readable handles and typed summaries. Prefer named functions, small records and literal outcome unions over deep generics, serialized JSON payloads, callback factories or user-assembled capability objects. Unavailable operations should identify the permitted next action in one local diagnostic. Show one recommended spelling in model prompts; old spellings are rewritten only in migration tooling; no runtime aliases are shipped.

### 3.3 Proposed folder iteration and proposal semantics

The recommended API is a folder receiver using the existing iteration scheduler, with explicit speculative edits:

```ts
// Proposed API. The host supplies a private target folder and a restricted evaluator.
const result = await targetFolder
  .iterateOn(improveStep, initialState, policy)
  .checkProgress('off')
  .withLimit({ maxSteps: policy.maxExperiments })
  .until(state => state.done);

return result; // { folder: immutable selected source, state: checked final state }
```

`improveStep` is a directory reducer with conceptual signature `(folder: Folder, state: SearchState, policy: Policy) -> SearchState`. It sees an isolated writable folder and returns only next state. Unlike callable-owned iteration, the folder operator carries two channels: committed source and typed state. `.until` receives read-only state and optionally a read-only folder view, so ordinary folder tasks can stop on file properties. Compiler inference supplies these callback types without author-written generic arguments.

A short authored step can express the improvement decision:

```ts
// Proposed API; durable operation IDs and tickets are supplied by exact helpers.
const proposal = await folder.propose(rewriteProgram, state.feedback, policy);
const report = await evaluator.evaluate(proposal.folder, requests.next(state));
const decision = selection.decide(state, report);
if (decision.accept) await folder.accept(proposal);
return advance(state, decision, report);
```

Required semantics and implementation tasks:

1. `folder.apply(reducer, ...args)` runs an edit and retains its selected changes on successful completion. `folder.propose(reducer, ...args)` runs the same reducer privately and returns a `FolderProposal<R>` containing a read-only immutable `folder`, typed `value`, base revision and inspectable diff. It retains no edits in the caller. Host-created identity fields cannot be forged by model-produced records.
2. `folder.accept(proposal)` installs exactly that proposal after checking folder ownership, base revision, edit policy and active transaction. A failed or stale acceptance changes nothing. A proposal is immutable; repairing it produces a new proposal and requires fresh evidence. Acceptance inside an iteration step remains staged until that step completes.
3. Each successful step atomically selects its folder changes and checked next state as one logical checkpoint. Failure, blocked/failed completion or result-type rejection advances neither. Durable publication uses one journal commit referencing immutable source and state; recover folder contents from that checkpoint rather than promising atomicity across arbitrary filesystems. External effects and paid evaluations remain recorded even if the step fails.
4. The receiving folder is not mutated by the whole iteration. Iteration works on its own branch and returns `{ folder: FolderSnapshot, state: S }`. Adopting the result into a caller or checkout is a separate explicit action. Nested iteration and nested proposal acceptance retain the same ownership rules.
5. `commit` include/exclude selection cannot cause source/evidence mismatch. The improver accepts and commits the exact evaluated projection; reject a commit that changes that projection, or require evaluation of the new projection before promotion. Record candidate source identity in both state and evidence.
6. Check the initial predicate before any edit; a zero-step result returns the original snapshot. Limits and errors carry the last joint checkpoint; retain the existing error distinction rather than reporting limit exhaustion as success. Normal experiment-budget exhaustion is an authored terminal state, with finalization capacity reserved separately.
7. Progress and observers see immutable source revisions and checked states. Hash source manifests rather than mutable Folder object identity. Observers cannot accept proposals or mutate the iteration. Provide explicit `'off'`, exact and semantic review policies; make semantic review policy explicit.
8. Do not hold a receiver write lock throughout inference/evaluation. Use private drafts, immutable captures, and a short checked publication operation. Snapshotting includes the active draft's selected contents without opening another write transaction. Exercise nested edits and concurrent sibling proposals for deadlocks and stale-base rejection.
9. Detect an empty source diff in exact code and return an explicit no-change proposal outcome. The authored policy decides whether more evidence is useful; do not spend a fresh target rollout merely to rediscover identical source when complete cached evidence is available. Invocation success, proposal acceptance, candidate promotion and final source adoption are distinct trace events with plain descriptions.
10. `folder.select(snapshot)` installs an immutable selected source from another population branch into the current iteration draft. This differs from accepting a proposal relative to its parent. Require a host-created snapshot belonging to the same authorized target, validate its full source projection against edit/contract policy, and use the destination revision captured by the active step as the publication expectation. Keep candidate-parent lineage and destination revision separate. Selection stages the exact snapshot; the existing joint source/state journal commit checks the expected destination revision and publishes both together. Exact authored selection helpers decide eligibility using evidence for that snapshot; the folder method does not choose a winner. A stale destination or foreign-target snapshot publishes neither source nor next state. Reuse existing snapshot and journal machinery; no new branch manager or unchecked patch rebasing is needed.

The simple incumbent-improvement example is not GEPA parity. The authored component-search program owns the experiment sequence, with exact helpers maintaining its candidate graph, seeded parent selection, protected incumbent and population. Exact helpers open the chosen parent's private draft, evaluate its child and update the graph. A child may enter the population without becoming the incumbent; then the iteration's selected folder stays unchanged. When the exact selector chooses another eligible candidate, call `folder.select(selected.folder)` and return state naming that same snapshot. Never install a foreign-parent proposal with `folder.accept` against the incumbent or equate "last edited folder" with "best program." The existing joint commit preserves source/state agreement.

Exercise incumbent `I`, non-incumbent parent `P`, and child `C`: `C` improves on `P` and enters the population, but `I` stays selected and returned source is `I`. Then independently select `C` and verify exact source publication; inject a stale destination revision and verify neither source nor state advances. Include source/state consistency after resume. Add these cases to the API migration/new-generation matrix under section 3.6.

Remove direct reducer calls from the model-facing surface: a direct call that succeeds but discards edits is an easy mistake for weak models. Prefer `folder.apply` for retained edits and `folder.propose` for speculative edits. Add a precise diagnostic and mechanical migration for direct `reducer(folder, ...)` calls; migrate existing callers and training data directly to the explicit replacement forms in this implementation. Reconcile `spec/SPEC.md`, `docs/ITERATE_ON_PLAN.md`, compiler intrinsics, prompts, authoring/integration skills, Node/browser APIs and teacher traces in that cutover. Folder iteration must reuse one scheduler and invocation kernel, not introduce a parallel interpreter.

### 3.4 Source as a value, evaluation as a supplied effect

Use one source authority for all target edits. A mutable target folder produces an immutable `FolderSnapshot`; a `ProgramSnapshot` adds a pinned entrypoint, external contract and execution-policy reference. Compilation yields a `CheckedBuild` tied to that snapshot and compiler identity. Evaluation yields evidence tied to the exact checked build, suite, model, seeds and scorer. These are inspectable host-backed values with durable reconstruction records, not returned closures or serialized live interpreter frames.

Expose the small-model facade `evaluator.evaluate(sourceFolder, request)`: it captures immutable source, validates the editable scope and execution policy, compiles and executes in fresh target contexts. Return a small discriminated report with `status: 'evaluated' | 'invalid-program' | 'infrastructure-failure' | 'budget-exhausted'`, source identity, approved feedback and evidence reference. Expose `evaluator.check(sourceFolder)` for bounded compile-repair workflows and a lower-level checked-build API for exact orchestrators. Models need not manually manage compiler products for the common case. Neither method chooses candidates or decides acceptance.

The evaluator is injected from the top as a declared typed service, not ambient `eval(sourceString)`. It closes over the pinned suite, allowed capabilities, accounting and cancellation. Candidate execution does not inherit the evaluator, builder, journal, optimizer services or improver captures. Assert this property across all target entrypoints and imported helpers; separate workers alone do not establish it. Compiled target exports never get installed into the improver's active callable namespace.

Implement instruction edits and file edits against the same snapshot-backed source representation. Inventory views and callable-site editors project onto that source. Treat authored source as canonical; derive overlays as an optimization and recompile the exact source projection. Return an `ImprovementResult` with selected source, final state, evidence references, source diff and disposition (`improved`, `baseline-retained`, `no-eligible-promotion`, `interrupted`, or `infrastructure-failure`). Returning a snapshot is an ordinary result; applying it to a checkout or deploying it is distinct. Package a complete source manifest so returned code is useful without a live run store.

### 3.5 Restricted execution and finite evaluation staging

Specify the intended language property before claiming Turing incompleteness. Unbounded `iterateOn` with arbitrary state can simulate general computation; a progress judge and a no-recursion rule do not prove termination. The present finite-loop/no-recursion checks remain useful restrictions, but are not a proof for the whole language.

Implement **mechanical termination with semantic guidance**, rather than instruction gas. Add `iterateOn(...).withMeasure(state => remainingWork)` and the same affordance on `folder.iterateOn`. Check a nonnegative safe integer measure at entry and require strict decrease after every step before publishing its checkpoint. A stalled/increased/invalid measure fails with the last checked source/state; zero remaining work before the goal yields an explicit exhausted disposition. The semantic progress judge remains active independently, can end an unproductive search early, and cannot replenish remaining work. Use remaining experiments for source improvement, not monotone quality: exploration may temporarily reduce quality. Keep `.withLimit` for explicit workflow step/deadline bounds; remove instruction/function-entry gas and shared generic execution-step pools.

Numeric counter loops capture a finite bound at entry and check strictly advancing counters, including IEEE-754 rounding failures. Collection loops traverse a finite entry domain so mutation through aliases cannot extend a Map/Set iterator indefinitely. Keep recursion/dynamic-code restrictions. Bound model turns, compile repairs and evaluation attempts through existing assignment accounting, and retain timeout/failure outcomes for foreign effects. Check all editable executable modules and declare native imports/effects outside the checked core.

S2 must publish the concrete terminating-core contract: finite values/domains, strictly decreasing iteration measures, recursion/dynamic-code rules, declared foreign effects and finite evaluation levels. Explain termination through finite domains and well-founded measures, assuming individual foreign operations return or fail. Unmeasured semantic iteration is a monitored host effect, not evidence of a terminating pure core. Update authoring examples, inference/types, training generators and existing traces directly; replay modified traces before admission. Generate cases for decreasing work with plateauing quality, semantic early stopping, semantic continuation at exhaustion, invalid/nondecreasing measures, checkpoint rollback, nested finite workflows, numeric rounding and alias-mutated collections. Research existing failure artifacts plus finite-worklist/search/task-planning sources to generate additional realistic cases.
Flat evaluation means ordinary targets receive no program-evaluation capability. Meta-evaluation uses a finite host-issued level: an outer evaluator may execute a frozen improver whose evaluator runs targets, but each evaluation reduces the permitted level. Reject attempts to mint evaluators, regain a level through new source identities, or use imported services/captures to recover parent authority. Freeze an improver copy for each run and adopt a transformed copy only between runs. The evaluation service is an explicit foreign effect; it does not add unrestricted dynamic execution to model eval or the checked core.

### 3.6 Training-data migration, source research and new generation for every API change

Use [DATA_LINEAGE.md](DATA_LINEAGE.md), [the task-source conversion plan](DIRECTORY_TASK_SOURCES.md), [the inline training-data plan](../docs/INLINE_NATLANG_TRAINING_DATA_PLAN.md) and [the implemented curriculum](../docs/INLINE_CURRICULUM.md) as the existing pipeline contracts. Extend their inventory, adapters, native replay, collection, admission and rendering paths; do not create an improver-only dataset bypass. Distinguish interpreter training (using tools and reacting to observations) from authoring training (writing TS/NL programs), and provide both wherever a change affects both.

#### Change registry and inventory

For every API change, create a versioned registry entry recording old/new contracts, affected source/task families, migration rule, source-research task, new-generation family, runtime freeze, coverage requirements and executable validation. Include changes introduced by all required capability rows and T1–T6, not only the initially selected folder methods. A host-only change with no model-visible consequence may record a justified no-data-impact disposition, but still requires an inventory scan; changed service declarations, observations or errors count as model-visible.

Default to broad automatic rewriting, not case-by-case preservation of the old surface. Related API changes share one registry entry, corpus scan, source-search report and generation batch when they exercise the same workflow. Rewrite authored examples, reference solutions, prompts, scaffolding and API-dependent expected results together, then validate the new behavior. Existing formatting, decomposition or wording is not an invariant. Preserve underlying task intent and independent behavioral checks, but update interface-specific assertions to the new contract. Archived originals and replacement lineage are enough for rollback; routine migrations require no individual approval. Use replay and recollection to resolve uncertainty before escalating a case for manual review.

Scan canonical case builders, source adapters, `natlang.program/2` IR, reference solutions, saved teacher jobs, native action/result traces, correction/preference pairs, static manifests and already rendered SFT. Include tool declarations, system prompts, embedded snippets, expected files/results and negative examples. Use `training/data_sources.json` and `data/teacher/data-inventory/current.json` plus the snapshot ledgers to account for active and historical inputs. Emit counts and affected record IDs by format/family, including indirect uses hidden in callbacks or generated code. A text search is a discovery aid, not proof of semantic compatibility.

#### Direct migration and verified replacement

1. Update the canonical builders and acquisition/conversion adapters first so future generation uses the selected API. Add versioned AST/schema-aware migrators for existing source, IR, reference actions and typed metadata. Migrate portable records directly rather than throwing away applicable training data or requiring a new teacher call for every spelling change.
2. Distinguish mechanical rewrites from semantic changes. For example, a direct reducer call that intentionally discards edits must become a non-accepting proposal whose typed value is used; it must not be rewritten to `apply`. A call whose task requires retained edits needs an explicit application and independently verified files. Folder iteration return shapes, private-branch behavior, state checks and progress policies need corresponding changes to callers and assertions.
3. Replay migrated records under the new frozen runtime, regenerate runtime-produced declarations, observations, source hashes, state/patch identities and evidence links, and verify original task outcomes with independent oracles. Reuse teacher decisions/reasoning only where they remain valid for the new semantics. Never attach rewritten actions to old execution receipts or invent observations to make a migrated trace appear executed. Record deterministic adapter/wrapper steps as such; do not relabel them as teacher reasoning.
4. When a full old trace cannot be faithfully migrated, automatically rewrite its usable task/source/oracle and enqueue recollection of the affected trajectory. Prefer a fresh verified replacement over leaving useful tasks indefinitely held. Review manually only when independent task intent, source rights or oracle meaning cannot be resolved by execution. Failed/obsolete positive examples do not automatically become preference negatives; regenerate and validate correction/preference pairs against the new context and causal checks.
5. Rerender existing SFT from migrated canonical records under the intended student template/tokenizer; do not patch serialized chat text as the authoritative migration. Regenerate sibling correction/preference artifacts and retain parent/split grouping. Audit prompts, declarations, observations and positive target actions for obsolete spellings; old spellings may appear only in explicitly labelled diagnostic-recovery or historical-context cases.
6. Write versioned replacement bundles, record old/new record IDs and hashes, migrator/runtime identities, outcome equivalence or deliberate semantic changes, and unsupported/recollection reasons. Update policy replacements and both default training entry points (`scripts/build_lora_sft.sh` and `scripts/create_training_pipeline.py`) to select the admitted replacements. Preserve originals as lineage, while directly updating the active dataset through those replacements. Pin active generation workers to their original freeze; new workers use the new API, and their results undergo the appropriate migration/admission check.

#### Source research and newly executed examples

For each registry entry, search both the existing local source inventory and external primary repositories/dataset releases for tasks that naturally require the new behavior. Deliver a source report with inspected samples, revision/checksum, license and repository rights, original split, available oracle/environment, conversion limitations and measured portable yield. Existing source-backed code edits, commit/refactor tasks, workflow trajectories and finite-domain property tasks are starting leads, not presumed coverage. An unrelated dataset or one API name in a prompt does not establish a useful source.

Keep research lightweight and executable: one shared report may cover several API changes, and an inspected existing source can be reused with a new adapter rather than rediscovered. Time-box external discovery and start native synthetic generation immediately for runtime-specific behaviors. Lack of a suitable external dataset does not block the API; a documented search plus independently checked generated tasks satisfies that portion of the work.

Implement bounded acquisition samples and acquisition/adaptation recipes for viable sources. Preserve actual task semantics: reconstruct needed base files and independent checks; do not translate shell/install/test calls into fake evaluator successes. Import source trajectories only when the necessary execution can be replayed. Otherwise use source tasks to collect new native trajectories. If no external source supports an affordance such as stale acceptance or evaluation-level rejection, record the search and construct exact synthetic fixtures with an independent oracle and explicit synthetic provenance.

Generate new examples that require observing the changed API's real output before deciding. Use the teacher collector against the frozen implemented API, inspect returned source/state/evidence, and admit only verified outcomes. Create counterfactual pairs whose visible opening matches but evaluation feedback, base revision or budget differs, requiring different next actions. Include complete short programs and individual decision contexts; cover nested composition without relying only on long optimizer runs. Synthetic gold actions or masked wrapper steps remain distinct from model-generated reasoning under the existing admission policy.

#### Required coverage matrix

| Changed affordance | Existing-data migration target | New source search and generation coverage |
| --- | --- | --- |
| Folder iteration and return shape | Callable/free iteration examples, directory reducers, TS authoring snippets and state/trajectory assertions | Multi-step edit/refactor tasks; zero steps, accepted and rejected edits, selected versus last attempt, private branch versus caller contents, nested iteration and joint checkpoint recovery. |
| `apply` / `propose` / `accept` and direct-call retirement | Direct reducer calls, folder tools, expected file patches and typed child results | Source-backed patch tasks plus exact speculative fixtures; value-only use, retained edits, no-change, stale/foreign proposal, failed child, selective commit and diagnostic recovery. |
| Cross-branch `select` | Population parent/child selection, source materialization and returned-incumbent examples | Non-incumbent parent with an accepted unselected child; later snapshot selection, stale destination, foreign-target rejection and resumed joint source/state consistency. |
| Snapshots, site editing and returned programs | `edit_code`/file-edit cases, template segments/captures, source export and installed-consumer examples | Instruction clarification and helper extraction sources; snapshot immutability, multi-file imports, capture/slot rejection, evaluation/source mismatch and fresh-consumer reproduction. |
| Evaluation/check facade and typed outcomes | Existing checker/service calls, follow-up cases and raw text/JSON proposal handling | Independently testable code repairs; compile diagnostics then bounded repair, evaluation failure versus target failure, paired accept/reject, and forbidden scorer edits. |
| Review policies, limits and finite staging | Existing progress-judge/iteration material, loop/recursion negatives and service scopes | Long useful progress, exact/off review, depleted shared allowance, nested allocation, bounded app TS and attempts to regain evaluator authority. |
| Evidence paging, roles, budgets and durable effects | Truncated feedback, service declarations, trace citations, accounting and retry examples | Observation-heavy workflow sources; decisive late evidence, invalid citations, unavailable role/capability, resume from stored results and unknown external effects without false retry guarantees. |
| SDK/CLI, adoption, inspection and diagnostic APIs | Existing authoring/integration programs and command examples | Runnable embedding/export tasks; explicit adoption versus returned code, stale checkout hashes, read-only inspection and local recovery diagnostics. |
| Transformation contracts and reusable reducers (T1/T2) | Edit-policy, helper repair and inline-site families | Refactor/clarification sources; verified versus empirical claims, no-change, scope violations, joint caller updates and deployment-model specialization. |
| Implement/repair/simplify and counterexamples (T3/T4) | Existing coding/failure/property cases and regression metadata | Contract-backed tasks with crisp checks; accumulate and deduplicate counterexamples, suite revision/cache changes, contradictory oracle and suggested versus admitted tests. |
| Portfolios and frozen-copy self-improvement (T5/T6) | Scoped specialist/routing cases and optimizer meta-cases | Independently grouped program families; fallback/routing failures, full-composition checks, frozen child searches, between-run adoption and finite evaluation depth. |

Extend this matrix whenever the selected alternative introduces an integrated experiment helper, program-value syntax, workflow builder or any later API. A removed API also needs coverage of the supported replacement and the diagnostic/migration path.

#### Admission and migration verification

Predeclare per-affordance positive, counterfactual, boundary/error and recovery coverage counts before generation; record admitted unique counts and rejected/unsupported counts rather than raw generation totals. Set actual family quotas after the bounded acquisition samples and before scaled collection. Split by underlying task/repository/program family, source lineage and generated world/seed; migrated and regenerated siblings stay together. Keep API-selection and final regression tasks evaluation-only. Compare initial API learnability separately from results after equal training-data exposure so familiarity with the old API does not silently determine the winner.

Run canonical native replay, oracle/source-integrity checks, lineage/split isolation, rendered-pair deduplication, intended-template/token audits and the existing joint-corpus reducer-share check. Update coverage/admission tooling so structurally valid examples that omit the required behavior cannot satisfy an API's coverage quota. No obsolete positive action may remain in the selected training inputs without an explicit held/historical disposition. Required existing inputs cannot disappear without a replacement or documented policy decision.

Every API work package completes only with: an affected-data report; admitted migrated replacements or justified non-applicability; a primary-source research report and bounded conversion sample (or documented unavailable-source result); verified new examples meeting its quotas; and updated default manifests/recipes. Publish a small-model regression probe over untouched tasks using the new declarations. Execute the required student training rounds within this assignment using pinned data/model configurations and declared compute allocations. Collection and training remain separate reproducible commands, not separate implementation assignments or permission checkpoints.

Migrate active affected inputs and generate verified positive, rejection and recovery coverage across the complete API. Automatically reconstruct/recollect incompatible but eligible examples and resolve the declared-snapshot curriculum coverage within this same assignment. Preserve lineage and evidenced exclusions without requiring historical trajectories to remain unchanged. Keep checks executable and batched; do not add a minimum viable release, per-example approval or paperwork phase. Process smaller batches internally and continue through the full coverage requirement.

## 4. Typed semantic differentiation

The useful analogue of prompt differentiation is evidence-driven discrete improvement: inspect where a program failed, form a hypothesis about an instruction, change it, then measure the effect. It is not an analytical gradient, and a plausible explanation is not causal evidence.

Proposed data contracts, checked before state changes:

```ts
type EvidenceCitation = {
  evidenceId: string;
  caseId: string;
  callId: string | null;
  component: string | null;
};

type Diagnosis = {
  hypotheses: {
    components: string[];
    problem: string;
    predictedEffect: string;
    citations: EvidenceCitation[];
    uncertainty: 'low' | 'medium' | 'high';
  }[];
  recommendation: 'rewrite' | 'collect-training-evidence' | 'no-change';
};

type ProposedValue =
  | { kind: 'lambda.instructions'; segments: string[] }
  | { kind: 'program.guidance'; text: string };

type InstructionProposal = {
  baseCandidateId: string;
  updates: { component: string; value: ProposedValue }[];
  rationale: string;
  citations: EvidenceCitation[];
};
```

Slots are taken from the descriptor by exact code; the model returns only replacement static segments. Validate selected keys, segment counts, finite data, exact capture contracts, citation membership and compiled source projections. Required rationale/citations improve reviewability without granting them evidentiary authority over scores. Empty/no-change proposals are explicit outcomes, not malformed JSON retries.

`diagnose.nl` examines local and end-to-end evidence, distinguishing wrong interpretation, missed helper, unavailable capability, expected blocking and infrastructure failure. Infrastructure faults route to an explicit recovery outcome; they do not motivate instruction rewriting as if the target answered incorrectly.

`propose.nl` returns one typed atomic update group. Give it current text, immutable contracts, representative training evidence and the diagnosis. Name capture bindings explicitly; provide exact signatures and access paths. For small interpreters, separate diagnosis from rewriting only when that extra call demonstrably helps; support a combined typed proposal function as a lower-cost policy.

Measure a proposal against its parent on paired cases/seeds before full validation. Implement bounded one-component ablations and counterfactuals as required experiment types available to the adaptive planner; the planner need not use them in every run. Never infer attribution solely from the model's hypothesis. Exact code enforces allowances, dependency groups and case availability. Generated tests expand training coverage through T4's independent admission and suite-versioning protocol; they do not silently become trusted validation/test oracles.

## 5. Search policies and exact lifecycle

Ship both strategies as required authored applications:

1. **GEPA/reflection:** preserve current seeded selection, dependency groups, conservative merges, minibatches, strict paired acceptance, protected baseline/incumbent, population bounds and validation selection. Replace raw reflection generation with the typed natlang proposer. This isolates the effect of the port.
2. **Adaptive planning:** `plan-experiment.nl` selects a bounded action such as rewrite an eligible update group, investigate a training failure, run an approved ablation/counterfactual, propose a semantic merge or stop. TypeScript validates and executes the choice; final eligibility and holdout rules are unchanged. Require independently tested evidence gathering, conflict resolution and at least one multi-attempt experiment-selection trajectory. This is a different strategy, not claimed to be GEPA-equivalent. Authored GEPA remains an explicit search strategy in the single implementation; adaptive semantic search is a required production capability.

Keep the per-case winner frontier and numeric acceptance exact. A semantic merge can propose a resolution where the existing conservative merge returns a conflict, but becomes a new candidate requiring the same validation. It cannot bypass acceptance by being labeled a merge.

Use explicit durable phases inside each authored search step:

```text
prepare -> baseline -> choose -> gather -> propose -> compile
        -> paired-evaluation -> full-evaluation -> commit-decision
        -> choose ... -> freeze -> holdout -> finalize
```

Every phase either returns the next serializable operation state, a terminal result, or a typed failed/interrupted operation. A search iteration completes with a candidate decision and a joint source/search-state checkpoint; phase transitions are operation events, not artificial semantic iterations. Baseline initialization has phases too, so a crash between training and validation does not force repetition of completed paid work. Retain invalid proposals and diagnostic attempts as events with usage and causes.

Use an exact stop predicate for completed selection, exhausted budget, requested cancellation or an explicit accepted stop decision. The authored `iterateOn` program owns repetition. Its default stochastic progress judge must not silently change search behavior or introduce unaccounted requests. Supply an explicit off/exact progress policy in compatibility mode; any semantic progress review in adaptive mode has a declared model role, allowance and persisted verdict. Do not claim an in-memory iteration trajectory is a durable checkpoint.

### 5.1 Durable semantic calls and effects

Persist semantic results, not interpreter REPL scopes. An optimizer cannot recover a previous native frame by loading JSON: captures, handles and partially executed evals are not portable continuations.

Journal an operation ID before dispatching model or evaluator work. Persist its inputs, source/model identities, seed, status and result reference. If a result is durable but its subsequent state commit failed, resume consumes the stored result rather than regenerating it. If a provider request was outstanding when the process died, mark incurred usage unknown; a retry is a new attempt with a new request ID. Exactly-once external model execution cannot be promised when the provider lacks idempotency or result retrieval.

Commit accepted next states with an expected journal revision; reject out-of-order or duplicate commits. Cache successful evaluations only under complete candidate/build/suite/profile/policy/seed/replicate identities. Retry only declared infrastructure classes, preserve failed-attempt ledgers and never retry ambiguous external target effects in shared state. Fresh disposable fixtures are the default.

Fingerprint the optimizer source and artifact, its service contracts, all model routes, target build, suite/scorer, engine policy and settings. Freeze optimizer instructions for the run. Old engine checkpoints resume under their original engine; migration creates a new attributed run or uses an explicitly tested schema migration, not a forced hash override.

## 6. Runtime and tooling extensions

Every listed capability is required in this implementation. The table is a scope checklist, not a priority queue of releases. Work according to actual dependencies and continue all available work without phase boundaries. Prefer small shared APIs to improver-specific exceptions.

Every row that changes an API or model-visible behavior includes section 3.6's migration, source-research, generation and admission work in this assignment. Runtime implementation alone is insufficient: migrate the applicable active training inputs and generate admitted data exercising the new affordance without deferring it to a later task.

| Scope | Extension/tweak | Required behavior and developer value |
| --- | --- | --- |
| Required | API/data change registry and migration pipeline | Inventory every changed contract; directly migrate applicable canonical examples/traces and active training bundles; replay/recollect, research sources, generate missing behavior coverage and update both default training recipes under section 3.6. |
| Required | Failure-driven improvement curriculum | Implement section 13's complete failure index, replay-to-case adapter and outer improver collection; generate measured source-rewrite/accept/reject trajectories rather than only local eval corrections. |
| Required | Small-model API experiments | Continuously probe the committed folder surface and at most two evidence-driven alternatives; freeze one contract and produce short examples/diagnostics without blocking on every hypothetical alternative. |
| Required | Folder-owned iteration and speculative edits | Implement section 3.3's joint source/state checkpoint, `propose`/`accept`, explicit review policy, typed return and source-aware trajectory over the existing scheduler. |
| Required | Source snapshots and evaluation facade | Implement immutable source values, source-backed site editing, `evaluate(folder, request)` and `check(folder)` with host-supplied flat authority; return selected source as well as artifacts. |
| Required | Bounded generated-program profile | Enforce finite allocations and evaluation levels; check all editable executable modules and declare native effects. Publish the actual guarantees without an unsupported termination claim. |
| Required | Typed improvement service adapter | Present current compiler/evaluator/store capabilities as documented declarations with bounded operations and host-enforced phase/split restrictions. Developers write natlang against normal services. |
| Required | Model role routing and accounting | Route improver functions separately from the deployment target executor and scoring judge. Record selected role/profile on each invocation/request. Start with one improver model per runtime; add declared per-component routes where multiple models are needed, with identity checks and fallback recorded. |
| Required | Hierarchical budgets | Extend current role accounting to include optimizer interpretation, diagnosis/planning, repair and progress review. Share a global ledger with parent/child scopes; sub-budgets allocate capacity rather than double-charge usage. Reserve finalization/holdout capacity. Pricing/bounds must be role-specific when models differ. |
| Required | Durable operation journal | Intent/result/state-commit primitives with stable IDs, revisions, attempt records, abort and honest unknown in-flight usage. Start at application step boundaries; arbitrary persistent REPL continuations are unnecessary. |
| Required | Typed candidate builder | Accept structured segment/guidance updates, assemble immutable candidates, compile projections and expose diagnostics without asking the model to encode JSON within strings. |
| Required | Paged evidence views | Filter by candidate/case/component, retrieve full observations and relevant trace sections with stable cursors/references. Replace the current fixed first-12 feedback slice and raw character truncation with explicit context selection. Keep complete durable evidence. |
| Required | Separate optimizer and target bindings | Trace and validate program ownership in both runtimes, including guidance and per-program adaptations. Service-triggered evaluation must preserve shared accounting without inheriting the caller's task frame. |
| Required | Engine identity/provenance | Register the authored engine/version and distinguish old native strategy, natlang compatibility strategy and adaptive planning. Cache and resume include optimizer build/model identities. Target artifact provenance records optimizer identity without treating it as the target executor. |
| Required | Typed progress and run inspection | Expose state phase, current operation, candidate lineage, instruction diffs, check changes, uncertainty, coverage, remaining budget and actual role spend through SDK and CLI events. A human explanation cites verified records and cannot overwrite factual report fields. |
| Required | Scaffolding and diagnostics | Generate a suite and scoped service declaration from compiler inventory; highlight missing coverage, undefined observations, capture/slot violations and infeasible allowances before paid work. Scaffold assertions as TODOs requiring author-provided semantics, not invented oracles. |
| Required | Debug/replay without paid inference | Inspect saved inputs/outputs and replay completed strategy decisions; distinguish historical evidence playback from fresh evaluation. Support a deterministic scripted semantic driver and a single-step operation command. |
| Required | Prompt/context inspection | Show the actual improver invocation text, contracts, evidence selection, compaction events and model route; export a reproducible operation bundle with sensitive host data handled by the application. |
| Required | Iteration inspection and persisted summaries | Build on the explicit review policy; expose requests/events, persist bounded summaries and use journal references for older states. |
| Required | Reusable exact algorithm libraries | Package selector/Pareto/dependency/statistics helpers for normal compiled applications with notices and tests, avoiding copied private imports into `dist/`. |
| Required | Validated source-change candidates | Implement the full capability: compile immutable virtual/worktree snapshots and compare declared external contract/capability changes. Store source edit proposals, not only instruction values. |
| Required | Cross-build evaluation | Implement the full capability: evaluate multiple builds against one pinned external suite/scorer without importing mutable candidate-local tests. Track component mappings and new inventory separately. |
| Required | Atomic source adoption | Required export and explicit multi-file adoption APIs with expected hashes, create/delete paths, build verification and rollback metadata. Users choose when to adopt; implementing and testing adoption is mandatory. Deployment remains a distinct action. |
| Required | Optimizer meta-evaluation | Required I/T6 deliverable: evaluate optimizer instructions and source transformations across independent target-program families; freeze each child engine and adopt versions between runs. |

Prefer familiar TypeScript calls and `.nl` functions for this implementation, but treat API and compiler refactors as first-class work. Do not force small models to assemble adapters to preserve existing semantics. New syntax or a workflow representation needs measured benefits over the short folder API. Introduce shared primitives rather than improver-specific exceptions; retire confusing old spellings through the explicit migration check.

## 7. Structural program improvement

Develop the bounded source-edit slice before instruction-mode parity; release general `source` mode after its production checks. Its request declares editable paths, allowed edit classes, public entrypoint/result contract, capability policy and the fixed external evaluation suite. Apply the bounded generated-program profile to editable application modules as well as callable folders when that profile is requested.

Proposals are structured edits with base build/file hashes. Support compiler-resolved node/site references for instruction changes and full-file/create/delete edits for restructuring; raw substring replacement cannot reliably describe changes to imports, types and helper folders. The host validates path containment, expected revisions and policy before compiling an immutable snapshot. Ordinary TypeScript and named/inline natlang functions may change, but test data, scoring, service implementations and the improver remain outside editable scope.

Candidate compilation creates a new descriptor and build identity. Preserve the external application contract by default. Internal captures/helper signatures may change together with their callers; import/type checks, callable-folder loop/recursion policy, capability declarations and regression suites verify the resulting program. Explicitly classify changed capabilities and public contracts as outside a behavior-preserving refactor unless the request allows them.

Keep the suite module/scorer fixed and feed each candidate through a separate entrypoint/snapshot handle. The existing instruction-only `PreparedSuite` and candidate artifact flow need extension here; revalidating a changed contract through the current overlay API is insufficient. Candidate-local checks can add evidence, but cannot redefine success or substitute for the fixed independent scorer.

Use a separate versioned source-change artifact containing base/new build IDs, complete source manifest, edit lineage, external contract comparison and evidence references. Produce a patch/private worktree for review; do not activate it as an instruction overlay. After adopting the new source, build and evaluate it normally, then issue a compatible instruction artifact if one is still desired.

Start with bounded helper extraction and instruction-plus-orchestration refactors. General dependency installation, arbitrary shell commands, provider configuration edits and repository-wide changes are separate capabilities, not implicit consequences of permitting source edits.

## 8. Developer workflows

Keep current evaluate/inspect/activate/export commands usable. Proposed additional UX:

```sh
natlang improve init ./project --entry main.ts          # author a suite scaffold
natlang improve ./project/suite.ts --dry-run
natlang improve ./project/suite.ts --engine natlang --strategy gepa
natlang improve ./project/suite.ts --engine natlang --strategy adaptive
natlang improve inspect ./runs/search --candidate ID
natlang improve step ./runs/search                     # one durable operation
natlang improve resume ./runs/search
natlang improve export ./runs/search --out ./review.patch
natlang improve ./project/suite.ts --mode source --policy ./edits.json
```

These are design examples, not existing CLI flags. Decide whether `improve` is an alias/facade over `optimize` during implementation; avoid two incompatible run stores or duplicated evaluation semantics. `resume` should find the recorded suite/engine or explain exactly which missing reference must be supplied. Maintain the established invalid-input, regression, infrastructure and interruption exit distinctions.

Proposed SDK surface: `createImprovementHost()`, `runImprovement()`, `resumeImprovement()` and typed async progress events. The host owns model sessions and service lifetimes; the application executes through `createNatlangRuntime` and its compiled `main`. Expose the optimizer's source path/version so authors can copy or extend it as a normal program without monkey-patching a private reflection prompt.

Expose both instruction and source modes in the completed implementation and make an authored strategy the SDK/CLI default. Users can explicitly select GEPA or adaptive planning. Implement default selection and run its section 10 measurements within this same assignment, without a separate graduation decision. Display the proposed work and estimated request envelope before execution when useful; estimates never override actual usage limits. Preserve supplied model choices and authorization. Results distinguish a selected improvement, retained baseline, no eligible promotion, interrupted incumbent and infrastructure failure.

Update both authoring and integration skills: the former covers writing objectives, diagnosing instructions and extending the improver; the latter covers embedding hosts, services, role accounting and durable lifetimes. Ship runnable packaged examples and a small complete improver program. Migrate their corresponding authoring/interpreter training examples and generate additional embedding/CLI tasks under section 3.6; copied documentation snippets alone are insufficient coverage. Documentation covers the complete implemented instruction/source and meta-improvement workflows, with proposed text replaced by the actual delivered contracts.

## 9. Complete implementation checklist

All rows are tasks in one continuous assignment, not sequential stages, intermediate releases or approval points. IDs are cross-reference labels only. Complete every row and T1–T6, including execution of data collection, both student training rounds and independent evaluation; do not stop at a generated recipe or partial runtime port.

| Work package | Deliverable | Required verification / evidence |
| --- | --- | --- |
| S0. API selection | Committed folder API with continuous small-model integration probes | Publish API mistakes, completion, integrity and spend; freeze one implementable contract, with at most two alternatives motivated by observed failures. |
| S1. Edit/evaluate/return integration | Folder proposal/acceptance, joint iteration state, snapshot evaluator and returned source | Instruction edit and helper extraction compile; accept/reject/no-change/repair/stale-base/zero-step cases pass; returned source reproduces evidence. |
| S2. Restricted execution and migration | Checked-core specification/termination argument or explicit reconciled claim revision; finite allowances, staging, compiler inference and diagnostics | Resolve the language-property decision in section 3.5; editable app modules cannot bypass the profile, targets cannot regain parent evaluator authority, and examples/data use the selected contract. |
| S3. Training-data migration and expansion | Per-API inventory/registry, direct canonical migrations, replay/recollection, source research/adapters and new native authoring/interpreter cases | Section 3.6 check passes for every S1/S2 API; active recipes select admitted replacements, all required behaviors meet predeclared quotas, and held-out probes remain untouched. |
| A. Extract boundaries | Reusable program/evaluator/store primitives, typed operation schema and immutable split policy | Existing native engine tests still pass; services cannot expose test inputs or invoke the old search loop. |
| B. Typed natlang proposer | `propose.nl`, candidate assembly and scripted fixture | Named/inline/guidance updates validate; captures/slots/keys are preserved; blocks, invalid proposals and repairs have durable typed outcomes. |
| C. Diagnosis and evidence | `diagnose.nl`, citations, pagination and context assembler | Large evidence is retrievable without silent truncation; invalid citations reject; failure causes stay distinct. |
| D. Authored GEPA engine | Port full GEPA/reflection orchestration and exact helpers into the application | All lifecycle decisions use authored code; fixed scripted proposals/scores match old engine selection/merge/acceptance behavior where intentionally preserved. |
| E. Durable operations/accounting | Phase journal, optimizer roles, sub-budgets and crash recovery | Crashes before/after requests, result writes and state commits resume correctly; no double usage accounting or false exactly-once claims. |
| F. Complete installed workflow | CLI/SDK engine registration, packaging, reports and source distribution | An installed package performs instruction, structural and adaptive lifecycles; deployment remains lightweight/compatible and the data migration/coverage checks are complete. |
| G. Measured adaptive improvement | Native versus authored GEPA/adaptive comparison, required semantic planning/conflict resolution | Execute section 10's independent target-quality/reliability measurements at matched spend; automatically fix defects and publish actual results, failures and overhead without pausing other work. |
| H. Structural mode | Immutable snapshot builder, cross-build evaluation and source-change artifact | Internal refactors compile and preserve external behavior; edits cannot mutate the scorer; stale patch adoption rejects. |
| I. Optimizer meta-improvement | Independent meta-tasks, frozen optimizer artifacts and T6 source transformations | Instructions and source can improve through the same system; between-run adoption/rollback work, and section 10's independent meta-benefit experiment executes with actual results reported. |
| J0. Failure-to-improvement corpus | Complete historical failure index and up to 120 unique incident-derived cases, using section 13 | Every discovered incident has a route; real outer improver trajectories exercise proposal/evaluation/selection and return reproducible source; corpus mix and spend are published. |
| J1. Continuing failure-driven generation | Incremental ingestion, automatic case/recollection queues, authoring/interpreter exports and verified preference pairs | Separate inventory and curriculum checks in section 13.7 pass; every eligible reproducible cluster has an executable case or evidenced reconstruction disposition, plus verified coverage or explicit exhausted-attempt status, preserving source splits. |
| K. Two student learning rounds | Failure-derived data -> trained small model -> untouched evaluation -> new training-pool failures -> second correction round | Both rounds execute with pinned models/data; section 10's student-learning measurements execute on independent cohorts with no evaluation-to-train leakage; report actual learning benefit and unresolved objectives. |
| L. Authored default and complete product | Authored default engine, production structural mode and all T1–T6 applications | All target/meta/learning experiments execute and results are reported; H/I/J1/K and T1–T6 are implemented and verified; all valuable native behavior is ported and the superseded native orchestration is removed. |

Follow real dependencies while continuing independent work: target evaluation needs implemented snapshot contracts, generation needs an executable frozen runtime, and round-2 training needs round-1 outputs. These are data/code dependencies within this assignment, not reasons to stop the task. Keep integration examples and checks running while completing the whole system. Every API task includes its corresponding section 3.6 migration/source-research/generation work.

Use recorded or isolated pre-refactor baseline results for comparisons; ship one authored SDK/CLI implementation with no legacy engines, compatibility modes or fallback selectors. Implement and verify that default in this effort. Measurement failures create automatic correction, replay or recollection work, without suspending structural features, meta-improvement, training or other remaining implementation. Do not add release waves, sign-off meetings, approval prompts or a separate decision to continue.

The assignment is complete in scope, not unlimited in inference/GPU spending. At setup, record one immutable experiment allocation in the existing budget/recipe manifest: total collection attempts, improver/target model requests and tokens/cost, target case executions, training executions/updates/GPU-hours, and development/confirmation evaluation counts. Populate finite numeric values from the declared workload and available authorized compute; this is configuration, not a new review or approval stage. Reserve capacity for both learning rounds and final confirmation before development draws from it. All retries, child searches, repairs and newly created batches consume the same allocation; batches cannot mint more capacity. Use existing ledgers and counters, not a new resource-management subsystem.

Default collection allowance is three attempts per planned unique failure cluster, each under section 13.6's case limits. Include a finite reserved count for new round-1 failures when computing the planned cluster total. Allow four planned training executions (two rounds and their matched controls), at most one additional correction/control pair, and at most two retries per declared infrastructure operation, all inside the same token/cost/GPU-hour ceilings. Predeclare development evaluation counts; allow one final confirmation execution. Resume does not reset counters or train-update limits. Reservations include unknown external usage conservatively until reconciled.

When a cluster exhausts its three collection attempts, record `exhausted-without-admitted-trajectory` with its case, attempts and observations; do not loop until the teacher succeeds. At any assignment-wide experimental ceiling, stop dispatching that category of paid work, record `assignment-allocation-exhausted`, and continue all remaining implementation, tests and inspections that fit their allocations. Report unmet coverage/learning objectives and actual external blockers honestly. Exhaustion cannot excuse omitted APIs, applications, adapters or tests, fabricate accepted data, or silently allocate another experiment. Keep the complete pipeline implemented and resumable for a later explicitly funded run without making that later run a requirement to finish this implementation assignment.

## 10. Verification and empirical acceptance

### Exact tests

- Same authored algorithm, scripted proposals and seeds produce preserved parent/component choices, compatible merges, paired acceptance and final selection. Deliberate new behavior has separate named tests.
- Named functions, authored inline sites with multiple slots/live captures, local helper metrics, guidance scope, compaction and concurrent target bindings remain covered.
- An improver call evaluating a target with the same relative function names does not trigger accidental recursion or leak adaptations. Target code edits cannot modify optimizer source or worker-shared fixture state.
- Training-only feedback and split-restricted evidence services hold even when asked for validation/test text. Frozen selection rejects later proposals; failed holdout does not reselect.
- Every model request, repair, nested `.nl`, progress review and judge is charged exactly once globally and attributed to its role/sub-run. Multiple model prices and concurrent reservations are tested.
- Failure injection covers every durable boundary, interrupted provider work, stale locks, duplicate operations, corrupt checkpoints and completed results awaiting a state commit.
- Candidate compilation rejects malformed segments, changed instruction contracts, stale source hashes and unsupported capability additions. Structural mode validates the new build rather than pretending contracts stayed fixed.
- Installed source and declarations resolve; vendor notices remain; optimizer code and journals do not enter browser inference assets.
- Folder iteration returns immutable selected source and checked state from the same checkpoint; rejected edits never appear in that source. Type failures, stale acceptances and failed commits preserve the last checkpoint; external effects remain journaled.
- Nested proposals, concurrent sibling edits, selective commits and active-draft snapshotting do not deadlock or execute a different source projection from the one scored. Limits and observer failures report the correct joint checkpoint.
- Returned source, base manifest and patch reproduce the selected checked build in a fresh installed consumer without a live run store. Overlay projections agree with their authored source form.
- Targets cannot recover evaluator authority via imports, callbacks, captures or newly generated identities. Finite allocations cannot be replenished by nested iterations; editable application TS obeys the requested profile.
- API migrators preserve intentional value-only versus retained edits, split groups and source provenance; migrated actions replay against newly produced observations/receipts. Semantic changes trigger verified replacements or recollection rather than stale success labels.
- Every API registry entry has inventoried affected data, recorded migration outcomes, source-research evidence and admitted new coverage. Active manifests/recipes select current replacements; obsolete positive actions, unsupported records and duplicated migrated/regenerated siblings cannot silently enter training.

### Live experiment design

Reuse triage, explicit-helper moderation and folder/stateful fixtures as a lifecycle baseline. Add several independently authored program families before measuring general improvement; three tiny suites are insufficient to select a broadly useful optimizer.

Compare frozen pre-refactor baseline results and the authored GEPA engine with the same target executor, paired seeds/splits and matched total spend allowances. Count optimizer interpretation overhead, not only target rollouts. Compare the adaptive engine separately. Include lambda-only, guidance-only and joint selections; check evidence requires at least two fresh paired baseline/finalist replicates for stochastic execution and reports uncertainty/sample sizes. Pilot sizing may use smaller samples. Repeated validation is a predeclared measurement protocol, not another selection opportunity on the locked test.

Report target quality/checks, helper coverage, paired changes, valid-proposal rate, infrastructure failures, incumbent/promotion status, model requests/tokens/cost by role, elapsed time and completed operations. Measure context efficiency and diagnosis citation correctness separately from target quality. Save/load a selected or explicitly retained-baseline artifact through installed Node and actual browser consumers.

The current LFM2.5 350M runs demonstrated lifecycle behavior and substantial failure, not instruction improvement. Exercise that model with typed proposals as part of the implemented small-model probe suite; fix API/curriculum weaknesses and continue the full assignment regardless of this model-specific outcome. A separately configured stronger optimizer model is a valid comparison, but its extra spend and identity must be explicit; the target executor remains the intended deployment model.

Complete the target, meta-improvement and student-learning experiments alongside the full implementation, pursuing the objectives below and recording actual outcomes. Individual runtime searches retain the baseline when no candidate is eligible. Measurement work never creates a milestone at which implementation stops or a reason to defer other required features.

### Required empirical measurements and improvement objectives

These experiments support the native test case. The primary implementation assessment is whether the whole improvement procedure is clearly expressible, inspectable, editable and executable as ordinary natlang, and whether small-model mistakes led to useful generic system refinements. Optimize that expression and its teachability rather than treating significance or winning against the old engine as the central project. Produce short runnable examples, inspect actual model contexts/traces, remove unnecessary helper plumbing and document concrete compiler/runtime/API refinements with before/after examples.

Before collecting measurement evidence, pin models/checkpoints, source families, scorers, case counts, independent splits, primary metrics, uncertainty methods, spend envelopes and permitted reliability margins. Use initial sample evidence to size cohorts, then lock the protocol before the final evaluation. A positive point estimate alone is not sufficient: the predeclared paired uncertainty analysis must support a positive effect. Report all runs, including retained baselines, invalid programs and timeouts; no post-hoc family exclusion or successful-case-only scoring.

Use development cohorts for iterative design and learning, and reserve **one final confirmation cohort** with target, meta-task and student-learning blocks. Freeze the system, checkpoints, comparators, sample sizes and metrics before opening it. Execute confirmation once for the whole assignment and report negative/inconclusive results without another fresh confirmation search. Compare starting, round-1 and round-2 checkpoints on the same final student block. Family breakdowns are descriptive; use a predeclared family-balanced aggregate as the primary target claim. If reporting formal significance for multiple primary claims, use a straightforward correction such as Holm across those claims with the declared threshold. Do not implement a sequential-testing service or spend time building a statistical framework. After confirmation, development can continue within remaining allocation, but revised versions cannot inherit the confirmed version's claim or reopen confirmation in this assignment.

1. **Target improvement:** evaluate at least six independent target-program families, including instruction/evidence reasoning, file transformations and structural helper/orchestration repair. Measure quality change over unchanged targets across all families, aiming for gains in at least three including a structural family. Use the predeclared family-balanced aggregate for the formal target-improvement claim; individual-family changes are descriptive unless separately predeclared and corrected. Run at least two pinned deployment model configurations, including the intended small model; measure the small-model aggregate within the declared total spend envelope and report whether it improves. Record baseline/GEPA/adaptive comparisons separately; beating every reference strategy on every family is not required.
2. **Adaptive capability:** complete real multi-attempt searches that choose evidence gathering, rewriting, ablation/counterfactual and semantic conflict resolution in response to observations. Demonstrate independent target gains for the adaptive engine itself; a GEPA-engine gain cannot satisfy this check. Exact policy and acceptance checks remain enforced for every search.
3. **Meta-improvement:** transform a frozen improver's instructions and at least one source/helper/orchestration component, then evaluate the new engine on untouched meta-task families. Measure improvement in the predeclared child-search quality/reliability/cost objective without relaxing target scoring, promotion rules or child budgets. Adopt the measured version between runs and demonstrate rollback.
4. **Student learning:** execute K's two rounds on the intended underpowered coding model. Evaluate trained checkpoints against the starting checkpoint and a matched-data/spend control using the existing non-outer-loop correction curriculum. Measure complete-loop success and source/evidence/false-promotion errors against the starting checkpoint, and the second round's additional benefit in its predeclared primary metric; claim supported gains only from final confirmation. Demonstrate failure-derived outer-loop data's incremental benefit against the control. Fix floor/ceiling metrics before evaluation; use a challenging cohort rather than declaring a zero-error baseline improved.
5. **Authored default:** implement the authored engine as the delivered SDK/CLI default and migrate default-facing documentation and training examples in this assignment. Run lifecycle/source-integrity tests and matched-spend reliability/quality comparisons against frozen pre-refactor baseline evidence; correct defects automatically while continuing all remaining work. Use the predeclared quality/spend objective to configure the authored default, with one authored implementation and no compatibility/native selectors. No separate graduation or release decision is required.

An unsuccessful development measurement triggers automatic diagnosis, native-system refinement or correction/recollection within the assignment-wide allocation, while independent implementation continues. Final confirmation is not retried. Preserve prior results and the protocol; never use locked evaluation failures as training examples or fabricate a positive result. Complete the required measurements within their allocation and report supported improvements and unmet objectives separately from the completeness and quality of the native implementation. Do not pause or abandon required features because an experiment is inconclusive.

## 11. Improving the improver

Once the improver is an authored program, its diagnosis, proposal, planning, semantic merge and explanation sites are normal adaptation components. This is the principal payoff beyond replacing one prompt: researchers and application authors can measure and customize the improvement process with the same inventory, traces, fixtures and artifact APIs.

Define meta-cases as pinned target-program/suite pairs. A fixture runs a bounded child search and observes target held-out quality, valid-proposal rate, completed lifecycle and total spend. Use independent target-program families for training/validation/test; instruction paraphrases of the same target do not count as independent families. Meta-judges and explanations cannot alter target gold, budgets or child promotion rules.

Freeze the optimizer baseline/artifact for each child run. Execute parent and child in separate runtimes/workers and journals with shared aggregate accounting. Meta-evaluation can score a child's frozen final result but cannot expose held-out target details to the parent's proposal process. Aggregate-only leakage policy must be specified and tested; arbitrary child report access is not a safe default.

Use one explicit meta-improvement level initially and complete both instruction and source transformation under I/T6. Preserve the previous optimizer version, all decision evidence and rollback. Active runs stay frozen; adoption occurs between runs. Finite host-issued staging supports bounded higher levels without regained authority, but unlimited recursive self-modification is outside the checked execution model. Modifying the runtime, evaluator or promotion policy remains an ordinary separately verified engineering change.

## 12. Required semantic-programming work packages

Implement all T1–T6 within this assignment using the folder/source/evaluation primitives; each exposes an authored program and an independent suite. Resolve their actual dependencies without introducing release boundaries or stopping points. Small-model results change implementation details and prompt/data design, not the required scope.

Each T1–T6 package must also deliver its section 3.6 registry entries, direct migrations of applicable existing training families, targeted local/external source search, bounded acquisition/conversion samples and newly collected verified examples. Extend canonical builders and active recipes when the feature is ready; do not defer training-data work until all six extensions are implemented. The coverage matrix specifies starting families and behaviors, and package checks include their predeclared quotas and lineage/admission checks.

### T1. Semantic transformation contracts

Implement a typed `TransformationSpec` containing the requested behavior/change, editable paths/sites, permitted edit classes, external interface, capability constraints, mechanically checkable invariants, empirical checks and finite allowances. Provide a short author-facing form with host-resolved references; do not make models construct fingerprints or scorer modules. A transformation returns selected source, its claimed effect, an explanation and evidence references.

Build a contract checker that classifies each obligation as `verified`, `violated`, `empirically-supported`, or `unverified`, with the checking method and evidence attached. Compile/type/capability checks can verify their actual properties; passing example tests cannot certify semantic equivalence. Missing required verification blocks adoption eligibility. Separately record model claims, so a confident explanation cannot upgrade an obligation's status.

Deliver one instruction-clarification and one helper-extraction fixture, generated declaration docs, and CLI/SDK contract reports. Verify: both transformations obey edit scope, intentional interface/capability violations reject, and an untested behavioral claim remains unverified.

### T2. Reusable transformation reducers

Package independently callable directory reducers for `clarify-instructions`, `extract-helper`, `remove-duplicate-guidance`, and `specialize-for-model`. Each has a short typed request, a TransformationSpec, a private draft, a typed rationale and no implicit evaluation or promotion authority. The application composes reducers with `folder.propose` and the supplied evaluator. Extraction must update callers/imports together; specialization pins the target model and records that narrower scope in the result.

Give each reducer exact negative fixtures and independent behavioral cases, including no-change and failed compilation. Add compiler-resolved semantic-site editing so simple instruction edits do not require a weak model to rewrite full TypeScript templates. Compare reusable reducers with one general rewriter under matched total allowances; retain a combined function where decomposition adds cost without reliability benefit.

Verify: installable source and contracts are inspectable, each reducer can run outside the optimizer, composed edits reproduce their source manifest, and transformations do not claim score gains without evaluation.

### T3. Implement, repair and simplify applications

Ship three authored applications using the same iteration and evaluator infrastructure:

- `implement-behavior`: take an existing skeleton or bounded empty project plus author-supplied contract/tests; generate a checked implementation. Undefined expected behavior yields a typed request for information, not invented gold.
- `repair-program`: take independently supplied failure evidence and preserve pinned regressions while fixing the declared defect. Separate target failures from infrastructure faults.
- `simplify-program`: preserve required checks while reducing a predeclared measurable quantity such as duplicated sites, source size or evaluated request cost. Exact code measures the quantity; independent behavioral checks remain mandatory.

Provide one complete example and suite per application. Keep objectives and acceptance rules distinct rather than calling all three quality optimization. Use a shared result/export format and fixed scorer boundary. Verify: each produces reviewable code with evidence, accepts a valid change, rejects a regression, and returns unchanged source when no eligible change exists.

### T4. Counterexample-guided development

Add a supplied `counterexamples` capability that searches a declared finite input domain or runs a bounded independent property checker. It returns a training-visible failing input, observed result, expected property and provenance; it cannot rewrite the oracle. An authored loop proposes a repair and adds admitted counterexamples to a versioned regression set.

Host-own the accumulating set, deduplicate by input/property identity, cap growth and retrieval, and journal suite revisions. A suite change invalidates affected evaluation-cache identities. Recheck baseline/current candidate on the same admitted set before comparisons; do not compare scores from different suite revisions as if paired. Keep a separate pinned validation/holdout suite for final selection. Model-generated tests are suggestions until an independent author/checker admits them; held-out failures stay outside the repair feedback channel.

Deliver a small finite-domain fixture with a crisp property oracle, a successful two-counterexample repair trajectory, and a contradictory-oracle fixture that stops explicitly. Verify: admitted regressions remain enforced, duplicate examples consume no new slot, generated expectations never silently become trusted gold, and suite versioning prevents stale evidence reuse.

### T5. Program portfolios and authored routers

Extend source mode with a bounded set of specialized implementations and an authored router sharing one external contract. The application may propose extracting a specialist or changing routing instructions; exact policy caps specialists, capabilities, runtime calls and total spend. Every routed result is scored by the fixed independent suite. The router uses deployment-visible inputs only and cannot access case IDs, split labels or expected answers.

Measure router errors, specialist coverage, end-to-end quality, latency and total inference cost; compare with a single-program baseline at matched allowances. Include overlapping specialties and an unknown-input fallback. Freeze the router and specialists as one source manifest and evaluate that full composition, not separately scored pieces assembled without a final check.

Verify: the portfolio exports as an ordinary runnable program, fallback satisfies the public contract, no evaluation metadata reaches routing, and any quality/cost tradeoff is reported explicitly. Do not add portfolio complexity by default without measured value.

### T6. Frozen-copy self-improvement of the improver

Extend I from instruction adaptation to a bounded source transformation of a frozen improver copy. Pin the evaluator, runtime, scorer, promotion policy and child budgets outside editable scope. The outer host supplies a finite evaluation level and creates fresh parent/child journals. Meta-cases cover independent program families and score child-search completion, selected target performance, invalid proposals and total spend under predeclared rules.

Return a proposed next improver version, its source diff, independent evidence and rollback reference. Adoption occurs between runs; running frames and active evaluator authority never change. Implement helper extraction, diagnosis/planning and orchestration refactors with independent checks as part of the complete source transformation capability; do not defer broader declared edit classes to a future release. Restrict parent-visible child reports to the approved aggregates; do not leak target holdout examples or allow selecting repeatedly on locked families.

Verify: old and new improvers run against identical held-out meta-tasks with matched budgets; recursive evaluation level cannot be regained through new program identities; adoption leaves active runs pinned to the previous version; rollback restores a reproducible version. A non-improving result is a valid retained-baseline outcome.

## 13. Concrete failure-driven self-improvement training plan

### 13.1 Outcome and existing machinery

Turn all discovered failures in generated training data into an indexed source of **DSPy-style discrete program improvement tasks**: inspect training feedback, diagnose a likely program defect, edit instructions or code, execute a bounded experiment, compare against the parent, accept/reject, and return the selected program. This uses natlang's actual runtime and the folder iteration API, not DSPy wrappers or a synthetic service that returns an optimizer's precomputed answer.

Reuse [the student improvement loop](../docs/STUDENT_IMPROVEMENT_LOOP.md), `ts-host/scripts/audit-rejections.mjs`, `build-handoffs.mjs`, `inline-curriculum/corrections.mjs`, `build-preference-pairs.mjs`, `admission-dispositions.mjs`, and the [failure corpus](../ts-host/scripts/failure-corpus/README.md). Those paths already find/replay local failures and collect corrected continuations. Extend them to build **outer improvement programs** whose target source is a separate folder and whose evaluator has independently pinned checks. A teacher fixing one failed eval is useful correction data, but does not by itself exercise a program-improvement pipeline.

All new script names, tasks and schemas below are proposed. The implementation freezes the selected runtime/API before collecting their trajectories. Existing inventory/replay can start first; do not stamp a scripted mock of an unimplemented API as executed training data.

### 13.2 Enumerate every historical failure, including recovered failures

Scan inventory-listed sources under `runs/` and `data/teacher/`, including raw `.result.json`, `.error.json`, partial jobs, trajectory/action ledgers, static-bundle rejections, replay/materialization/admission/token audits, handoffs and corrected variants. Read full originals rather than only the truncated snippets in rejection reports. Extend the current rejection audit, which skips ultimately accepted rows, to index failed/rejected actions inside successful runs and their later repairs. Link any existing successful correction instead of paying to rediscover it.

Create an immutable `failure-incidents.jsonl` and incrementally refreshed queue. Each incident records: parent program/run/site, source artifact/hash, runtime/model/seed, original split/group, failure observation references, starting and effective source at the site, input/fixture references, applicable independent oracle, later correction references and processing disposition. Deduplicate copies and range exports by original run/site and content; separately cluster related causes without deleting occurrence counts. Classification is a routing hypothesis, not a model-generated causal fact.

Every incident gets one route: `program-improvement`, `decision-repair`, `api-migration-and-replay`, `infrastructure-recovery`, `dataset-oracle-repair`, `evaluation-only`, `duplicate-covered`, or `needs-source-reconstruction`. The index covers all failures; generation prioritizes distinct train-eligible clusters rather than producing an expensive duplicate for every occurrence. Counts must distinguish incidents, sites, clusters, cases and admitted decisions. Refresh from a pinned snapshot first, then ingest new completed artifacts by hash; partial jobs are updated when their durable results arrive.

| Observed failure | Case to generate | Required successful behavior |
| --- | --- | --- |
| Wrong interpretation, reasoning or missing evidence | Instruction/guidance improvement over the actual failing target | Cite observed training failures, change a semantic site, execute paired checks and retain only an eligible candidate. |
| Argument/return mismatch, capture/slot misuse, forbidden loop or recursion | Source/instruction repair plus check/evaluate | Read real diagnostics, edit source/callers, compile, measure and return a checked candidate; do not just retry a failed eval unchanged. |
| Incomplete file edits, wrong counts, missing clauses or malformed output | Directory-reducer improvement with return-value and file oracles | Improve both returned data and selected files; a file-only or answer-only pass is insufficient. |
| Missing observation, premature choice, unwarranted helper edits | Evidence/decision-policy improvement | Retrieve decisive evidence or preserve correct code; distinguish satisfying the task from merely appearing to use the desired technique. |
| Successful run with failed actions followed by a real fix | Recovered-failure improvement and shorter authored workflow | Use the existing fix as a candidate seed, then test its source-level generalization; do not assume a local fix is a global improvement. |
| Timeout, budget churn, repeated calls, context exhaustion | Bounded context/decomposition improvement or recovery | Measure avoidable spend/churn, paginate or decompose if supported, preserve quality; return no-change/recovery when the environment is the cause. |
| Provider/rate-limit/dependency fault or ambiguous external effect | Infrastructure-recovery control case | Diagnose accurately, use permitted recovery or stop, preserve baseline, and never claim instruction quality improved. |
| Old API/runtime, faulty or underspecified oracle, source mismatch | Migrate/reconstruct and replay, or dataset-repair lane | Automatically update API-dependent data/checks; fix a demonstrably bad oracle outside the target loop, then freeze it before creating a target-improvement case. |
| Candidate regression, stale proposal, false acceptance, evidence/source mismatch | Outer-improver repair/meta-case | Correct the improver's selection/edit procedure in a frozen copy, then independently execute a bounded child search. |

Do not treat a rejected trace as proof that its authored program is defective. Replay with the pinned target executor first. If failures are stochastic, retain that classification and use paired seeds/replicates; if a current baseline succeeds, route to decision repair or a recovered/API-migration case. Existing held-out incidents remain evaluation-only even when they are useful examples of an error category.

### 13.3 Construct an executable target and independent improvement suite

For each selected cluster:

1. Reconstruct the actual `.nl`/TS target, initial folder, service declarations, relevant fixtures and source revision from full program IR and trace evidence. Preserve the failed starting source separately from effective edits later made by the run. For eval-only failures, promote the reusable faulty logic into an authored helper/site with a declared contract, or create an instruction-policy task; record this transformation rather than pretending the historical source contained that helper.
2. Replay the original failing input on a pinned executor. Keep the full failure and any successful correction as host-side references. Expose only permitted training feedback to the improver; do not put a gold patch, the final selected program or hidden suite answers in its opening.
3. Build a small family suite using existing sibling tasks, source-backed reference checks and exact fixture generators: default to 3 training cases (including the incident), 2 selection-validation cases, and 2 locked final cases. Include regressions the old target already passes. Generate additional inputs broadly where a crisp/reference oracle exists; for semantic tasks use the existing independent scoring protocol or collect reviewed expected behavior once, outside the search. Lack of a reliable oracle routes to a reconstruction task, not indefinite review of an otherwise automatable migration.
4. Split the outer task by the full union of its target/source/correction lineages, before rendering; all derivatives stay together. Inner training/validation/final cases share an outer case's scope but obey its separate feedback restrictions. No outer-training task may contain a target or correction from an existing outer-held-out group. Locked outcomes are consumed only by finalization and admission, not fed back to proposals.
5. Supply a `TransformationSpec`, editable scope, baseline snapshot, pinned suite/scorer, deployment model and finite allowances. Begin with instruction/guidance rewrites; use bounded source mode for source defects. Checks reject editing gold/scorer/services or hardcoding example answers in place of the declared general behavior.
6. Register a `natlang.program/2` curriculum task, extending its case metadata as needed for source/evidence checks. Its root invokes the actual authored improver; typed services execute real candidate checks/evaluations. Its final oracle verifies selected source, evidence identity, disposition, required decision sequence and regressions. A retained baseline is correct when no eligible candidate exists; target quality gain is not necessary for every training case.

Represent the outer request with a small record: `target`, `feedback`, `policy` and an injected evaluator. Keep hashes, fixture handles and run-journal operations host-managed. Package target sources and external suite manifests alongside case IR so a fresh frozen-runtime consumer can reproduce the case without an old live session.

### 13.4 Teach the whole loop and its decision points

Collect two complementary views from each executable case:

- **End-to-end interpreter trajectories:** run `folder.iterateOn(improveStep, state, policy)` through diagnosis, proposal, real evaluation, comparison and selected-source return. Include two-attempt cases where the first edit fails or regresses and the second succeeds, plus all-rejected and no-change outcomes.
- **Focused continuation tasks:** hand off at diagnosis, first edit, compile repair, feedback interpretation, rejection, acceptance or final selection. Reconstruct the actual outer call state and prior receipts. Use existing handoff/replay infrastructure rather than fabricating an opening that claims a target was evaluated.

Add observation-driven counterfactual groups: identical visible task/source/proposal, different real fixture outcomes or base revisions, requiring accept versus reject, repair versus stop, or reevaluate versus stale refusal. Both variants execute their evaluator. Keep scripted proposals as explicitly labelled fixtures for teaching exact decision rules; separately collect naturally generated proposals so the corpus is not entirely scripted search.

Produce authored TS/NL program examples as a second lane: the target failure, requirements and contracts lead to a short improvement program using the supported API. Compile and execute each authored program on the same independent suite; do not admit source-only answers on plausibility. Include reusable diagnosis/rewrite reducers and concise exact-helper calls rather than making every example reimplement the journal or Pareto selector.

Keep three model identities explicit: the data teacher, the frozen improver interpreter being exercised, and the deployment target executor. A teacher can execute the improver for collection, but cannot substitute its own corrected answer for a target rollout. Count every diagnosis/rewrite/repair/judge call and target evaluation under the case's ledger.

### 13.5 Admission, SFT and preference outputs

Admit **outer improver decisions** only after the independently checked outer continuation completes with reproducible selected source. A successful evaluation request that reports a bad target candidate is valid context for a correct rejection; it is not a bad outer decision. Tag traces by program ownership so the materializer does not accidentally promote the failing target's inner decisions into SFT. Retain rejected candidate source and scores as observations; failed tool actions remain nonpositive context under the normal admission rules.

Verify paired parent/candidate outcomes, acceptance policy, source projection, allowed split access and selected-versus-last-attempted source. Rebuild the selected source in a fresh consumer and repeat its declared final checks. Do not require success on locked cases for admitting an honest no-promotion/failure report, but never label a final-check failure as an independently validated improvement. Keep reported target performance and correct optimizer behavior as separate fields.

Build preference pairs only from the same outer decision context with replay-proven alternatives: a valid edit over an invalid edit, a correct rejection over false acceptance, evidence collection over an unsupported conclusion, or returning the selected snapshot over a rejected last attempt. Reuse `build-preference-pairs.mjs` with an outer-program adapter and causal checks. Do not label one plausible rewrite worse merely because its prose differs; preference based on candidate quality requires paired independent evidence and the declared tie/uncertainty rule. Infrastructure and oracle faults are not target-reasoning negatives.

Export versioned `cases.train.ir.jsonl`, evaluation-only IR, `outer-trajectories.jsonl`, `authoring-programs.jsonl`, admitted interpreter/authoring turns, correction variants, causal preference pairs, and coverage/lineage/spend manifests. Run the standard student-template/token, deduplication, split and reducer-share audits. Preserve incident -> case -> search -> proposal -> evidence -> exported-decision links for every admitted row.

### 13.6 Generation tasks within the complete implementation

Implement and execute the following tasks under the normal resumable pipeline; suggested filenames live in `ts-host/scripts/self-improvement/` unless noted:

| Task / proposed script | Concrete output / integration |
| --- | --- |
| `index-failures.mjs` | Read inventory and existing rejection/job ledgers; write incident/cluster JSONL, route counts and missing-source queue. Extend the rejection scanner to include recovered failures. |
| `reconstruct-targets.mjs` | Batch API migration, source reconstruction and native replay; write baseline source bundles, replay evidence, linked existing repairs and automatic recollection queue. |
| `build-cases.mjs` | Create scoped target suites, outer train/eval groups, editable contracts, end-to-end and focused continuation IR; emit explicit counts by route/failure mechanism. |
| `collect-improvement` | Use the frozen collector plus actual compiled improver and target evaluation adapter; write durable outer jobs, requests, proposals, target evidence and selected-source results. |
| `verify-improvement.mjs` | Check outer lifecycle/source invariants and rebuild final programs; write admission and correction records. Adapt native materialization and preference replay to preserve outer/inner ownership. |
| `export-improvement` | Produce interpreter/authoring SFT and causal pairs via normal preparation/render/audit paths; update active replacement manifests and both training builders. |
| `ingest-next-failures.mjs` | Enqueue newly observed target and improver failures, linking duplicates and completed replacements; generate the next bounded collection batch. |

Use `runs/self-improvement-data/<runtime-api-version>/<batch>/` for reproducible intermediates and `data/teacher/self-improvement/<version>/` for admitted exports. Extend `scripts/create_training_pipeline.py` and `scripts/build_lora_sft.sh` with this data lane; make the complete recipe continue automatically from collection/export through both training rounds and evaluation, with resumable commands for each operation. Extend `create_student_improvement_pipeline.py` to consume these cases for the required K student probe -> outer handoff -> teacher correction -> training/evaluation rounds. Data collection has its own runnable endpoint, but collecting data without executing K does not complete this plan. Training runs use explicit pinned recipes and declared compute allocations; updating this plan does not launch them.

Internal generation batch sizing (continue automatically through all required coverage):

1. Index **all available historical artifacts**, without applying a collection cap to discovery. Publish route/cluster counts from that actual snapshot; do not reuse old rejection totals as current inventory.
2. Attempt up to **120 unique train-eligible incidents**: 60 instruction/evidence/semantic cases, 30 bounded source/file/type repairs, 15 recovered-failure/context-efficiency cases, and 15 honest-rejection/infrastructure/no-change controls. Cap the initial batch at 10 incidents per underlying program family. If a lane lacks real incidents, reallocate and publish the change; synthetic supplements are separately counted.
3. Start with a **12-case smoke batch** spanning at least six available mechanisms, including one measured acceptance, one regression rejection, one compile repair and one retained baseline. Then collect the remainder automatically from the ranked queue. Reserve an independent outer-evaluation probe from eligible original evaluation groups, never by repurposing their failures into train.
4. Default to at most **3 candidate attempts**, **24 improver model requests** and **40 target case executions** per case, with target per-invocation limits inherited from the frozen executor. All baselines, replicates, retries and finalization count. Reserve final-check capacity before search; reduce attempts or training cases if paired replicates need it. If the envelope is infeasible, emit a typed budget outcome and adjust the next batch's declared policy rather than silently enlarging the current case's allowance.
5. Target at least **24 admitted complete loops** in J0's expanded batch within the assignment allocation, covering positive improvement, rejection after measured regression, repair after compile feedback, no-change/baseline retention, and selected-source reproduction. Also export verified focused continuations and authored programs; report their actual unique counts rather than treating them as additional complete loops. If that target is missed, focus remaining allocated attempts on uncovered mechanisms, then report the coverage shortfall. Three failed collection attempts terminate that cluster with `exhausted-without-admitted-trajectory`; never admit failed data or create unlimited batches to force the count. Small batches are internal work units, not delivery stages.

Rank subsequent work by recurring failure mechanism, lack of current curriculum coverage, replayability and recovery cost. Use broad source/task rewrites and exact generators to generalize each real incident, while retaining an explicit connection to its historical failure. Deduplicate trivial paraphrases. Source search under section 3.6 is for additional examples of observed mechanisms, not a prerequisite to using the already available failure corpus.

### 13.7 Continuous coverage and completion

After each generation batch, every new failure updates the incident registry. Failed **targets** produce more target-improvement cases; failed **improver decisions** become replayable outer handoffs and, once I/T6 is ready, frozen-improver meta-cases. Automatically batch recollection and update admitted replacements. All batches share section 9's finite assignment allocation; a new batch draws remaining capacity and cannot replenish it. Exhausted clusters keep their explicit final disposition rather than returning indefinitely to the queue.

Use two distinct completion checks against a declared immutable discovery snapshot:

- **Inventory completion:** all historical artifacts in that snapshot have been scanned, every incident has a route and linked provenance, and missing reconstruction/recollection work is visible. A routed backlog satisfies inventory completion only.
- **Curriculum processing:** every distinct train-eligible cluster has an executable case or evidenced reconstruction limitation, and a terminal status: verified/admitted coverage, linked verified representative, evidenced source/split/oracle exclusion, `exhausted-without-admitted-trajectory`, or `assignment-allocation-exhausted`. Record actual admitted coverage separately; exhausting attempts is not evidence of mastery. Train eligibility follows source/split/oracle rules, not ease of collection. Held-out groups remain evaluation-only. No cluster stays in an endlessly replenished recollection queue.

J1 implements both checks and a working incremental ingestion path under the assignment allocation. Attempt J0's complete-loop target and report any admitted-coverage shortfall with terminal dispositions rather than making teacher success an unbounded completion condition. Publish a failure-mechanism coverage dashboard with admitted loops/decisions, remaining backlog, target quality change, false-promotion rate, retained baselines, evaluator/compile failures and total spend. Measure immediate API learnability on untouched development tasks and K's learning benefit using the single reserved final confirmation block. Process the declared snapshot and reserved new-failure workload within the same finite allocation; later corpus growth does not extend this assignment's spending or scope indefinitely.

### 13.8 Required two-round student learning experiment

Every change of target model requires a fresh training-pool diagnosis. Select semantic repair cases from observed failures of that model, and efficiency cases from its successful but expensive action traces; do not infer current failures from historical labels. Include bounded target actions/observations and public service declarations in training feedback. Keep validation/test outputs sealed. Expose measured modelCalls in the injected typed evaluator API. The native editor uses an exact diff-derived finishing helper, writes complete replacement text, and rejects no-op proposals independently of claimed changed paths. Ground hypotheses in real input fields and service operations; do not invent schema or replace semantic judgment with keyword matching. Compare parent and candidate on the same declared seed.

Port existing authored editor examples to the exact finishing helper and replay them against the current runtime. Recollect decisions that need newly available trace/schema feedback rather than fabricating observations in old data. Search generated failures for no-op edits, unsupported fields, redundant inspection/completion, failed action batches, fixture errors and target timeouts. Generate and independently verify fresh teacher corrections, regression rejections and retained-baseline cases for these mechanisms, with model identities, source-group lineage, real diffs and measured costs. These cases exercise the native improvement program itself; ordinary prompt rewrites with no executed edit/evaluation/selection trajectory do not substitute for them.

Execute both rounds as part of this same assignment, consuming audited J0/J1 exports as soon as they are available and continuing the rest of the implementation throughout:

1. Pin the intended small-model checkpoint, runtime/API, training recipe, tokenizer, train lineage registry and compute/spend limits. Freeze an independent multi-family outer-task evaluation protocol sized under section 10. Measure initial complete-loop success, correct candidate selection, returned-source reproducibility, false promotion, request cost and target held-out quality. Evaluation examples/answers never become collection seeds.
2. **Round 1:** train on admitted failure-derived outer-improvement interpreter and authoring data, mixed with the declared general coding/replay curriculum to preserve basic capabilities. Train a matched starting-checkpoint control with existing local correction data and equal optimization/token/compute allowance. Record actual unique examples and token exposure; the control does not receive outer-loop data.
3. Evaluate the resulting checkpoints on the untouched round-1 cohort, then probe the improved model on a separate train-eligible program pool. Feed its new failed target and improver decisions into J1's reconstruction, outer handoff, correction and preference collection. Use those real new failures to construct round-2 data; do not merely repeat round-1 examples or mine locked evaluation failures.
4. **Round 2:** train the round-1 checkpoint on audited new corrections with a declared replay mix. Continue the matched control with its corresponding local-correction lane and equal compute exposure. Evaluate on the untouched round-2 cohort; compare the starting, round-1 and round-2 checkpoints on the same cohort and apply the predeclared paired uncertainty method.
5. Publish both rounds' data lineage, model/checkpoint hashes, training settings, spend, complete-loop metrics, target quality, errors and control comparisons. Execute and report section 10's student-learning measurements as part of K. Use development outcomes and train-pool failures to refine the native API/curriculum within the remaining allocation. Perform only the allowed correction/control executions; final confirmation is evaluated once under section 10 and is never rerun in search of significance. Report negative findings and exhausted allocations as unmet empirical objectives while completing the whole native implementation.

This experiment requires actual training and independently evaluated checkpoints. A generated recipe, a collection-only export, a stronger teacher's success or a model's plausible explanation cannot substitute for those results.

## 14. Definition of done

An application author can install the package, inspect the improver's actual natlang source, configure its models and allowances, provide an independently scored target suite, run GEPA or adaptive planning in instruction/source mode, interrupt/resume, inspect every decision, adopt runnable selected code, deploy or reject the result, and extend the improver without editing private host internals. The delivered SDK/CLI uses the authored engine by default; implementation and measurements are completed in the same assignment.

Complete the entire checklist and T1–T6 in one implementation effort; there are no partial-release completion points. Assess the deliverable first as a natlang-native test case: provide the complete authored program, short ordinary-user examples and before/after evidence of shared API/compiler/runtime/tooling refinements that removed native-program awkwardness. Outcome measurements support that assessment; they do not replace it. All required capabilities and T1–T6 checks pass: structural program transformation, adaptive experiment selection/conflict resolution, reusable contract-backed reducers, implement/repair/simplify applications, counterexample-guided development, portfolios and measured self-improvement of frozen improver copies. The authored application uses `folder.iterateOn`; exact invariants, independent scoring, finite evaluation staging, separate program contexts and visible accounting survive every transformation. A small model edits, evaluates, accepts/rejects and returns selected source without assembling host bookkeeping.

Resolve the language-property decision, complete all API/data migrations and declared-snapshot curriculum processing, report achieved coverage and exhausted dispositions, execute source research and new generation, and run every target/meta-improvement experiment plus both actual student training/evaluation rounds on independent families under declared spend. Runnable returned code reproduces its checks; source adoption and between-run improver adoption/rollback work; default-facing SDK/CLI/docs/training data use the authored engine. Report measured benefit, failed hypotheses and unmet empirical objectives honestly, without turning them into pauses or deleting required scope. A wrapper, partial runtime, unaccounted failure backlog or recipes left unexecuted despite available allocated capacity do not satisfy this assignment. Explicit exhausted-attempt/allocation dispositions finish bounded experiment processing without pretending the missing coverage or learning benefit was achieved. Deliver the complete implementation, verification evidence and experimental results together in one final handoff.


Implementation constraint: this system has no deployed users. Rewrite the API and migrate callers, builders, tests and training data directly. Do not ship compatibility codecs, deprecated aliases, legacy engine switches or duplicate orchestration. Port every useful existing GEPA/reflection behavior into the single authored system before deleting superseded implementation; keep old behavior only as pinned offline comparison evidence where needed.

Implementation refinement: make folder revisions native values (`snapshot.digest`, `folder.at(id)`); remove model-managed repository registration. Service scopes use service-name keys and exact callable paths, with explicit `/**` inheritance. Require a decreasing work measure or finite workflow step bound for every iteration that actually steps. Migrate callers, scoped service metadata and recorded teacher actions directly; replay those translations before admission. Keep prior responses as historical evidence, not an active compatibility corpus.

## 15. Implementation handoff — 2026-10-01

See [implementation and experiment evidence](../docs/PROGRAM_IMPROVEMENT_RESULTS.md) and [user workflow](../docs/PROGRAM_IMPROVEMENT.md). This is one implementation effort, not a staged rollout. Runtime, authored GEPA, source editing/evaluation/return, source adoption, T1–T6 interfaces, failure-driven collection, migration and both actual student rounds with controls have been implemented and exercised. There are no compatibility APIs or legacy scheduler modes.

The default `improveStep.nl` calls the authored exact `lifecycle.step`, which measures the baseline, delegates semantic diagnosis and source editing to one `rewriteProgram` directory reducer, and installs the best measured population member. Retire model invocations that merely copy measurements or assemble bookkeeping: baseline and selection use exact authored helpers, not separate semantic reducers. Evaluator authority is granted to exact evidence-owning paths. This deliberately refines the shared native composition model for weak interpreters instead of pushing search into a host scheduler. A real development repair and source-size simplification executed this composition and passed independent replay. The SDK validates source/state identity, compilation/edit scope, exact iteration advancement and independent quality before publishing a checkpoint; invalid steps preserve the previous valid pair.

Canonical migration processed 118 files, 746,044 rows and 2,220 API edits. The current corpus manifest pins 225 admitted outer decision turns from 19 unique trajectories. Discovery and terminal incident statuses are pinned in `runs/self-improvement-data/folder-api-v1/completion`; exhausted dispositions do not constitute successful coverage. Source research artifacts remain labelled reference-only until executable reconstruction and independent verification can support training admission.

The complete suite passed 698 tests, 22 conformance cases, browser type checking and application builds. Final source/workflow checks and Python renderer/pipeline checks also passed. Training and serving share a fingerprinted history-preserving template policy; turn output now respects the SDK allowance. These last refinements occurred after confirmation and have development/test evidence, not a new final-cohort learning claim.

The empirical objectives remain unmet: the single final confirmation found 0/6 fully successful held-out loops for every student checkpoint, and frozen-improver confirmation had effect 0 and p-value 1. Both student rounds and both controls were genuinely trained. The intended 24 admitted loops have a five-loop shortfall, and the full curriculum matrix is not covered. The shared assignment has six reserved provider calls and 21 reserved case executions remaining, insufficient for another complete collection/reproduction pair; a final confirmation cannot be reallocated. No new batch replenishes that allocation. These bounded negative outcomes are recorded alongside the implemented native system; they are not silently replaced by teacher success or unexecuted recipes.


### Structural headroom and a simpler native loop

Use the native optimizer as a language design test, including its own unnecessary model calls. Keep hypothesis selection and source transformation semantic; put deterministic evaluation sequencing and acceptance in ordinary authored exact helpers. The root calls `lifecycle.step`; exact lifecycle code supplies source and training evidence directly; diagnosis returns a plain hypothesis string; the exact lifecycle invokes `experiment.test` once. Remove the old baseline/selection semantic reducers completely, without aliases. Keep finite `folder.iterateOn`, branch ownership, same-seed comparison, independent host checks and separate acceptance/selection. A failed experiment must lead to a distinct evidenced hypothesis, rather than repeatedly clarifying the same instructions.

Run structural experiments, not only instruction edits. Start with the actual per-review semantic delegation cost observed in Bonsai traces: batch independent semantic judgments into a typed array and aggregate exactly. Include negation, sarcasm and mixed sentiment to prevent keyword shortcuts. Prove training headroom with an independently authored reference hidden from Luna, compare a direct Luna rewrite against three native experiments, and freeze all sources/runtime before one final confirmation. Report every attempt and distinguish operational success, training headroom, validation selection and held-out effects. `ts-host/scripts/self-improvement/structural-study.mjs` implements this study and emits native training-case seeds.

Migrate existing optimizer demonstrations to the new helper calls and recollect authentic trajectories; do not retain obsolete reducer APIs in the runtime or fabricate helper observations in old turns. Search generated-data incident indexes and existing review/pagination traces for per-item delegation, redundant completion, no-op editing, fixture confusion and repeated rejected hypotheses. Add independently labeled batches plus native optimizer repair cases, retaining source groups. Replay and admit actual edit/evaluate/reject/select trajectories before SFT. Reference rewrites and locked confirmation outputs never enter optimizer training evidence.


### Atomic completion and one experiment per lifecycle call

Make `eval {code, finish:true}` the recommended completion action for weak interpreters. It returns the fresh value of the final expression or explicit return only after checking the declared result type and normal terminal invariants. A missing/invalid result or failed computation cannot publish an older staged answer. Keep ordinary inspection/staging when finish is omitted; this is one eval operation with explicit completion intent, not a compatibility mode. Exercise source writes plus typed completion, exact aggregation, wrong result types, missing results, older staged answers and failed effects in native/browser tests and training data.

Split semantic planning from deterministic experimental execution. `planExperiment` returns only `{hypothesis}`; its private folder is discarded and its scope does not include the experiment executor or editor. The lifecycle performs one private rewrite/check/train/validation/acceptance operation and exact independent population selection. A rejection is a result for the outer `folder.iterateOn`, not permission for the model to run extra experiments inside one iteration. Retain all valuable GEPA/frontier, acceptance-versus-selection, lineage, seed, checkpoint and allocation invariants without retired reducer aliases.

Treat the current .nl interpreter as the semantic worker. Avoid nested delegation for small visible semantic inputs; use exact code for aggregation. Shorten the editor prompt, explicitly separate inspections from finished results, and prohibit proxy target execution in the editing helper. Use serial benchmark arms on the shared student server and scale pilot case size/deadlines to complete real comparisons rather than labeling queue latency a semantic error. Retain failed and superseded development attempts and their allocations. Freeze one final cohort only after development across all source/runtime refinements concludes.

Migrate both authored optimizer snapshots and saved eval tool schemas. `migrate-improver-lifecycle.mjs` produces current templates/schema candidates, preserves historical observations, revokes stale admission and records recollection manifests. Existing approved canonical corpora stay auditable; new atomic actions come from actual current executions and independent replay, never fabricated merged tool calls. Search and generate cases for inspection values returned as RewriteResult, unnecessary nl calls by the editor, fresh final expressions completed atomically, stale-result completion after failure, and extra experiments attempted inside a single planner invocation.

### Improve the optimizer by supplying its working context

Make deterministic parent sampling and evidence assembly exact authored operations. GEPA samples the measured frontier with the pinned seed; adaptive search starts from its incumbent. Give the semantic planner an ExperimentContext containing the chosen parent, complete source text, public training observations, exact opportunity classification and prior experiment outcomes. Its sole result is a hypothesis string. Remove evaluator authority from the planner entirely. Supply sourceFiles in every RewriteRequest as well; the editor can combine its complete replacement and bookkeeping.finish in a single eval finish:true. Models should not spend calls discovering paths or reconstructing values already known to their caller.

Port implementation/repair/simplification adapters to the current editor path and exercise their actual execution, not merely wrapper construction. Migrate prior optimizer templates and RewriteRequest demonstrations directly; preserve historical observations and recollect affected actions. Generate current trajectories on observed semantic-delegation overhead, failed atomic completion, no-op edits and repeated experiments. Compare the prepared-context optimizer with the frozen preceding optimizer on the same closed development tasks, recording optimizer requests, tokens, valid experiments, student cost and completion. Reuse only exact baseline execution identities with original costs retained. This optimizer ablation does not reopen the preceding final test cohort and makes no new held-out claim.

Carry the latest actual candidate source and training execution trace into the next editor invocation, including compilation failures and rejected cost regressions. Keep only the latest detailed feedback in portable loop memory; history retains compact outcomes. Never expose validation answers as reflection feedback. Do not stop cost/size search at its first improvement: treat the improved incumbent as a new starting point, continuing until no further supported hypothesis or the declared finite experiments end. Exercise two successive cost reductions and a semantic stop in integration tests. Generate curriculum cases from original delegation overhead and the measured failed rewrite; give them fresh independently declared test cases, preserving the closed original confirmation cohort.


### Refine the shared semantic language through the optimizer

Make direct semantic answers the default when inputs are already visible. Share this guidance between general natlang calls and the small-model program opening: the current call performs judgment; delegation must add a distinct subproblem or handle genuinely large data. Strings return as plain text, other declared results as JSON. Exact computation and effects finish with a fresh typed `eval` result. Remove the separate planner invocation. Supply the diagnosis-and-edit reducer with its bounded evidence brief and complete source files; return a typed edit summary derived from the actual folder diff. Keep raw structured context available for targeted inspection and honest clipping; do not introduce optimizer-specific macros or hidden authority.

Display large records with a bounded per-field overview that preserves late fields and small scalar values. Name clipped fields precisely in scope and retain full values for targeted reads. Preserve complete small values rather than clipping them unnecessarily. Exercise this outside self-improvement with large irrelevant logs followed by a semantic task. Record matched student accuracy and requests on identical source and inputs, keeping reserved validation/test cases unopened.

Pair each generated action with the observation caused by that action. Capture terminal action results and use model-request event boundaries rather than millisecond timestamps. Test an error followed immediately by successful atomic completion: the optimizer must see the error on the failed action and the final value on the successful one.

Rewrite all affected authored callers and training templates directly. Revoke admission for changed executions and recollect, rather than rewriting historical observations. Search prior optimizer exchanges and canonical generated cases for repeated inspection, hidden late fields, redundant nested judgments and wrapped one-field results. Generate general boolean, string and structured semantic-result cases alongside actual source-editing loops. Require independent replay before positive SFT. Compare optimizer requests/tokens and valid drafts on pinned development observations; compiler-valid drafts alone are not student improvements. Preserve every attempt and the original final cohort.


### Ordinary runtime and native follow-up implementation

Use the ordinary shared runtime prompt and all ordinary tools for the optimizer. Remove optimizer-specific replacement prompts and tool filters. Apps may append `systemPrompt` instructions; the shared language guidance remains present. Keep improvement fixture execution in an explicitly selected application adapter, with a hashed identity in collection provenance; the generic collector contains no improvement-specific branch.

The default semantic composition is one diagnosis-and-edit directory reducer per experiment, called by the exact authored lifecycle. Preserve the separated planner/editor only as frozen comparison source, never as a runtime compatibility option. Port branch-parent selection, rejected candidate feedback, population selection, exact quality/cost checks and semantic stopping into this composition. Compare it with a direct rewrite on twelve independently specified semantic programs, including classification, extraction, pagination and multi-file reducers. Prove development headroom with a separate reference implementation hidden from the optimizer. Account for optimizer and student requests separately and together. Freeze selected sources before the one confirmation case per program; retain the declared descriptive/multiplicity rules and assignment allocation.

Optimize the authored optimizer itself on fresh, independent inner targets. Measure the full nested request count and expose compact causal diagnostics plus actual rejected source and training observations. Keep the flat evaluator injected only at the owning root; children inherit ordinary language tools and explicit lexical service scopes. Freeze active optimizer source; adopt edited source only between runs.

Fix general language issues revealed by this workflow: inline semantic definitions need deterministic, parent-qualified identities, preserving recursion rejection without treating independently created nested functions as recursion. Directly loading a child `.nl` must preserve the type scope it has through its owning callable folder. Exercise both outside the optimizer. Update compiler/evaluation fingerprints and reexecute affected training trajectories; preserve historical provider responses and annotate seed-identity migrations rather than presenting recorded responses as fresh model sampling.

Generate training from actual native loops. Reexecute both optimizer and student actions, require matching requests and complete recorded responses, and compare source, state, quality, cost and disposition. Exclude sealed confirmation cases from training replay and fixtures. Export each root/child invocation with its actual initial arguments, source root, typed result and before/after folder; independently replay child editing effects. Admit successful intermediate and terminal actions using trace model-request boundaries; retain failed actions as context for later corrections. Quarantine incomplete provider recordings and report them as curriculum gaps. Recollection uses a finite declared development allocation and fresh execution; it never repairs a recording by inventing a response.


### Follow-through from observed directory-program failures (2026-10-02)

Remove inaccurate runtime affordances directly: directory-reducer guidance must describe only implemented folder handles and offered tools, including exact computed file writes and typed completion. The nonexistent global `fs` helper is removed rather than implemented as a compatibility alias. The editor must distinguish source-edit paths from target runtime outputs and repair the execution method when trace evidence shows an unavailable API; restating the semantic goal is insufficient.

Run data migration against a pinned current SDK through both whole-loop replay and independent invocation export. Explicitly migrate recognized prompt contexts, public fixture declarations and uniquely recoverable source/case identities; preserve original provider replies, verify all remaining input/continuation content, and reexecute exact actions against independent value/file oracles. Preserve unsupported, interrupted and incomplete trajectories as failure curriculum; do not admit them merely because a typed loop state was returned. All 73 useful turns from 28 invocations in seven original loops have now been ported.

Generate and execute fresh source-editing cases from actual training failures, varying matching paths, file counts and irrelevant preserved files. The current refund/urgent/recommendation assignment uses Luna as optimizer and Bonsai as student with two train folders and one validation folder per program. It has one shared 400-call/96-case/four-hour allowance, including the documented fixture-correction attempt. Do not reuse prior held-out cases, replenish this allowance, or require empirical success before shipping the general language fixes. Replay/export the completed trajectories, report actual accepted-source behavior and cost, and retain every unmet empirical objective explicitly.

Completed evidence: all three new file programs improved to validation quality 1 through edits to their ordinary `solve.nl` directory reducers. The assignment used 143 provider calls and 40 case executions, including the fixture correction. All three loops replay exactly; 12 independently checked root/child invocations contribute 39 additional useful training turns. Probe observations are reused only for explicitly cached baselines; fresh executions retain their own observations. The second proposal round achieved no additional measured gain. No new confirmation cohort or weight-training run was performed.

### Rich directory programs and optimizer weight learning

Make the next integrated experiment a test of native source restructuring, not just prompt repair. Six fresh directory families start with executable, partly correct programs: final refund intent, invoice joins, exact text edits, latest review revisions, eligibility appeals and meeting confirmations. Use nested paths, unrelated files that must survive, typed JSON outputs and concrete output-file checks. Require independently supplied labels to execute the simple starting case before spending provider calls. Helpers belong under `solve/`; the optimizer can rewrite the root, remove a poor helper, introduce child callables or batch semantic judgment and aggregate exactly.

Give the editor the previous candidate's actual changed helper source and diagnostics at the front of its compact context. After a first repair, continue the declared native search for another distinct improvement. Measure rejections followed by successful correction, structural changes, successive gains and semantic stopping separately. Compare native iteration against one direct rewrite and a hidden reference. Freeze all sources and the SDK before the experiment; do not redefine reference timeouts as evidence of semantic headroom. Run every family without empirical success gates, within one 2,400-call/520-execution/six-hour assignment. Preserve fixture mistakes and their costs in the same allocation. Report exhausted or failed arms explicitly.

Strengthen training admission directly: independently successful child file edits are positive targets only when their measured source experiment was accepted. Rejected candidate edits remain original failure context, available to exercise later correction, rather than positive demonstrations. Reexport existing verified cohorts under this rule: the original 73 turns become 40 positive targets and the fresh file-repair 39 become 33. Preserve all original provider observations and historical manifests. Search actual generated failure traces for wrong joins, shallow file patterns, superseded decisions, whitespace loss, repeated rejected edits and inefficient semantic delegation; reconstruct fresh executable curriculum without sealed confirmation answers.

Train Sharp MiniCPM5 2B to execute the optimizer using these admitted Luna decisions and new development trajectories. Bonsai remains the unchanged target-program executor. Use the pinned BF16 MiniCPM source weights with the selected Sharp tokenizer/template, verified file hashes, completion-only rendering and a strict token audit. Run one logical 32-step LoRA pilot with four sequences per step, rank 8, learning rate 0.0001, seed 20261002 and maximum length 16,384. Resume only its own frozen-data checkpoint. Keep the owned DGX job within 16 GiB container memory and a 4.5% CUDA allocation; preserve other servers.

Automatically prepare, render, audit, train and compare starting/trained optimizer checkpoints after finite source collection. Exclude all transfer-family trajectories from training, even teacher successes. Freeze both checkpoint identities before one descriptive comparison on eligibility appeals and meeting confirmations. Evaluate correct selected source on independently labelled sealed cases, valid termination, false promotion and total optimizer/student requests. No revised cohort or tuning follows confirmation. Bound comparison at 600 calls, 128 case executions and four hours, within a twelve-hour distillation assignment. Exhaustion is an explicit outcome, not a reason to replenish the experiment.

Improve the general runtime and model transport where these executions expose weaknesses: context compaction must keep the latest action and observation usable even when reasoning alone exceeds the window; retain full reasoning in the transcript. Decode Sharp XML exact strings and typed booleans without executing text inside reasoning or filling omitted parameters. Exercise these independently of self-improvement and migrate/replay affected training contexts before admission. Implementation and current outcomes are tracked in `docs/NATIVE_DIRECTORY_CAMPAIGN.md`; running experiments are not evidence of learning gain.

Preserve independently verifiable work inside failed loops. If an interrupted target request prevents whole-loop replay, isolate a complete recorded editor invocation under the identical SDK and original artifact hash. Reexecute its exact inputs/actions, require the typed result and file effects to match the parent experiment's measured candidate, and apply accepted-source admission. Label this editor-only evidence explicitly; retain full-loop quarantine and rejected edit context. An absent target response is never reconstructed. This first directory run yields six actual rejected-then-corrected editor decisions alongside its independently verified unchanged terminal invocation.


### Expanded default optimizer training slate

Make native program improvement a default producer in the general training pipeline, rather than a separate experimental artifact. Generate 192 fresh tasks across 16 semantic and exact-bookkeeping families, each with three train folders and two validation folders. Exercise root/helper edits, adding child callables, correcting semantic evidence scope, preserving exact file effects, identity joins, revision selection, ranking, and further reduction of semantic delegation. Use independent latent labels and exact effects, hiding gold and reference source from execution. Do not reuse sealed evaluation inputs or mislabel reconstructed mechanisms as reproduced historical failures.

Execute every task through the ordinary authored directory-reducer improvement program with Luna as optimizer and Bonsai as target executor. Use three finite experiments per task; persist resumable source/allocation journals and every failed disposition. Allocate the whole collection explicitly: 192 tasks, 34,560 calls, 6,144 executions and 192 hours, with per-task ceilings of 180 calls, 32 executions and one hour. No successful-headroom probe or per-family pass condition gates later collection. Run one serial worker to respect shared serving resources.

Replay and independently admit actual trajectories against the collection SDK. Publish admitted turns incrementally into the shared content-addressed self-improvement registry, retaining other producers and historical receipts. Include both generation stages and their verified corpus output in every newly built default general training recipe. Retain failed/rejected trajectories as evidence for subsequent curriculum construction rather than positive demonstrations. Pin already running recipes and the separately declared Sharp distillation pilot. Implementation and live collection paths are documented in `docs/NATIVE_DIRECTORY_CAMPAIGN.md`.

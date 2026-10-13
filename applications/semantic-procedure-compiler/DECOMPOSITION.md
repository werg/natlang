# Semantic Procedure Compiler

## Behavior and design before implementation

Compile an immutable collection of evolving prose procedures into a typed callable natural-language
application. The compiler must resolve which editions govern, separate permissions from duties,
compose scoped exceptions, infer the carried state and dependency graph, lower semantic obligations
into named functions and invoked scoped literals, and exercise the generated procedure on connected
normal/adversarial scenarios. A well-typed generated folder is insufficient: scenario judgments must
show that its decisions implement the governing source meanings. Ambiguous source authority remains
an explicit compilation diagnostic rather than a fabricated branch.

The first world is a fictional inland radio club's station handover procedure. Older habit notes,
a signed replacement procedure, a local emergency appendix, separately owned equipment custody and
an ambiguous witness report interact. The first fixture will contain at least30 connected clauses and
six scenario histories, including a receipt that proves custody but not readiness, an emergency waiver
that removes a queue prerequisite without waiving a safety handover, an old same-named permission,
and a late return that changes the next operator's permitted action without retroactively authorizing
an earlier broadcast. External law and radio regulations have no role in this authored toy world.

## Distinct typed contributions and modules

| NL callable | Owned transformation | Consumers |
| --- | --- | --- |
| `compileProcedure.nl` | Folder reducer: join governed meanings, generation, scenario results and revisions | Final CompilePacket |
| `compileProcedure/intake.nl` | Classify original immutable records and bind compilation goal | Edition selection, scenario inventory |
| `compileProcedure/intake/interpretGoal.nl` | Interpret required observable behavior and compilation constraints | Intake.goal |
| `editions.nl` | Select governing documents by authority, replacement scope and event time | SemanticRequirementSet |
| `editions/selectAuthority.nl` | Judge issuer competence and actual adoption vs habit/draft | EditionSelection.authorityDecisions |
| `editions/resolveSupersession.nl` | Join changed clauses with preserved untouched clauses | EditionSelection.selectedClauses |
| `requirements.nl` | Construct source-grounded requirements, permissions and exceptions | ProcedureModel |
| `requirements/interpretRule.nl` | Read one obligation's actor, prerequisite, effect and scope | RuleMeaning[] |
| `requirements/resolveException.nl` | Scoped literal distinguishes waived prerequisite from surviving duty | RuleMeaning.exceptionBindings |
| `model.nl` | Join state schema, causal order and unresolved dependencies | LoweringContext |
| `model/inferState.nl` | Infer typed carried facts, provenance and unknown values from actual rule needs | StateSchema |
| `model/orderDependencies.nl` | Build partial source-backed transition graph | DependencyGraph |
| `model/orderDependencies/settleComponent.nl` | Settle one component and expose unresolved negative cycles | DependencyGraph.components |
| `lowering.nl` | Partition semantic responsibilities into callable modules and local judgments | ModulePlan[] |
| `lowering/chooseBoundary.nl` | Decide shared named stage vs contextual literal from actual reuse/ownership | ModulePlan.boundaries |
| `lowering/planBranch.nl` | Scope a condition's guard, effect and unknown branch | BranchPlan[] |
| `writing.nl` | Author numbered deep-NL module bodies from returned module plans | GeneratedProgram |
| `writing/writeModule.nl` | Emit one typed callable with algorithm, witness obligations and consumed children | GeneratedModule |
| `writing/writeModule/auditContribution.nl` | Detect forwarding-only stages and unused literal results semantically | ContributionAudit |
| `scenarios.nl` | Reduce independent scenario states through generated transitions | ScenarioSuiteResult |
| `scenarios/replayStep.nl` | Advance one ready transition and persist new typed state/proof | ScenarioFrontier |
| `scenarios/replayStep/judgeGuard.nl` | Invoke contextual NL guard over actual source/event state | GuardDecision |
| `scenarios/checkOutcome.nl` | Judge required/forbidden consequences and source-grounded uncertainties | ScenarioVerdict |
| `compileProcedure/reviseStep.nl` | Directory reducer: repair one demonstrated semantic defect on private candidate | RevisionState |
| `compileProcedure/reviseStep/explainCounterexample.nl` | Identify earliest rule/state divergence rather than patching a final label | RevisionTarget |
| `audit.nl` | Join interface, provenance, contribution and scenario obligations | CompilationAudit |
| `audit/checkInterfaces.nl` | Judge caller/child state ownership and preserved unresolved fields | InterfaceFinding[] |
| `compileProcedure/packageProgram.nl` | Return files, source-to-stage map, scenario certificates and remaining diagnostics | CompilePacket |

## Actual different call paths

- `compileProcedure → editions → selectAuthority → scoped authority judgment`: governing source selection.
- `compileProcedure → requirements → resolveException → scoped exception judgment`: a changed prerequisite
  set consumed by model ordering and scenario guard evaluation.
- `compileProcedure → model → orderDependencies → settleComponent`: finite graph-component settlement.
- `compileProcedure → writing → writeModule → auditContribution → scoped contribution judgment`:
  returned audit changes emitted module ownership or exposes a diagnostic.
- `compileProcedure → scenarios → replayStep → judgeGuard → scoped rule literal`: a scenario-specific
  guard result is consumed as the next state transition, not merely reported beside an unchanged result.
- `compileProcedure → reviseStep → explainCounterexample`: private folder candidate repair followed
  by full interface/source/scenario audit; existing edition/requirement analyses stay immutable.

Different guards capture different rule meanings, current proof frontier, actor identity and event
knowledge time. They are instantiated from source-selected rules and directly called on actual scenario
states. Source facts and derived status do not share an undifferentiated string map.

## Iteration and state

Edition selection consumes a finite document/clauses frontier. Dependency settlement consumes finite
strong components, preserving unresolved cycles instead of toggling truth. Module writing visits the
returned finite module plan once. Scenario replay uses `iterateOn` with unprocessed transition IDs as
its decreasing measure; each step either returns a new typed state or settles an unresolved blocker.
Revision consumes demonstrated counterexample IDs, and a source amendment is never invented to force
a scenario to pass. Private folder candidates use the existing directory reducer/Folder transaction;
no new durability mechanism is introduced.

## Exact seams and why they are crisp

- Load immutable file IDs, bytes and hashes: outside-world I/O only.
- Validate source quotations, generated relative paths and duplicate file IDs: exact identity checks.
- Parse/type-check emitted `.nl` signatures and callable folders with the existing compiler: exact
  language/interface mechanism. It cannot decide whether a typed branch matches a prose exception.
- Serialize generated files and reports / apply private bounded edits: exact plumbing and durability.
- Verify declared typed invariant IDs, scenario numeric timestamps and evidence references: external
  checker only. Semantic scope, selected rules, code boundaries and scenario entailment remain NL.

No TypeScript helper chooses an edition, applies an exception, decides a guard, computes a permission,
partitions modules, invents a counterexample repair or scores semantic usefulness. Generated policy is
numbered natural-language pseudocode in callable `.nl` files, not a crisp decision engine.

## Qualification and provenance

This decomposition precedes implementation. Target roughly28 authored `.nl` modules and600–900 lines of
substantive structured NL, plus a generated callable-folder witness. Scale is an architectural target,
not an admission limit. The task snapshot will bind visible original source files, request and scenario
state, source-world split identity, original fictional provenance and license. Oracle and complete typed
canonical witness remain outside visible roots. Independently reconstructed source invariants, actual
loader/type/reference replay, comparator positives/negatives and live literal/state-flow review are
separate gates. Source/replay preparation never grants training admission; model-facing changes require
roughly48 live samples per variant or recorded measurement debt.

Current source foundation:28 authored NL modules /620physical lines,33connected visible clauses,6scenario histories,5different consumed scoped judgment sites and30declaredcall edges. Parser/type checks pass. Full typed generated-program witness, independentoracle/comparator qualification and liveexecution remain pending.


## Independent-review correction: symbolic replay must expose generated divergence

Astra found that the first symbolic guard interpreter could read the ideal source model and silently
correct an incorrectly emitted guard. The revised guard stage invokes two independent scoped NL
judgments: actual emitted-body interpretation uses actual atoms and code text, while source-required
interpretation uses a separate sourceAtoms channel and governing meanings. Their permissions/effects
and actual local decisions have distinct typed fields; matchesSource is an explicit comparison.
Replay preserves the actual emitted effects even when wrong, and applies source-required effects only
to the independent reference state. Generated evidence uses typed path/quote spans separate from
original source Witness citations. The station-global hazard input belongs to ST; personal queue and
brief acknowledgment remain actor-owned.

This runner is symbolic NL interpretation of emitted bodies, not invocation of the generated named
callables. Exact parser/type validity and symbolic scenario certificates do not grant generated-runtime
execution credit. Actual generated-program invocation and live behavior remain separate qualification
work. Original fixture/oracle source facts are unchanged by this correction.

Revised source after peer findings:28NL /656physical lines; independent emitted/required judgments share one guardsite but return separate consumed contributions. Source parser/type and quickgate pass.

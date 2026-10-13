# Saga Forensics — application decomposition

Status: first authored slice implemented and source-loader checked; live execution and independent source/oracle review pending. Program ID: `astra-app-saga-forensics-v1`.

## Behavior and scope

Given an immutable directory of client journals, worker traces, receipt extracts, migration notices and contractual clauses, reconstruct which business effects actually stand. Distinguish a repeated request, a repeated acknowledgment and a genuinely separate effect. Derive the smallest justified repair policy, including observation branches where an outcome remains unknown. A repair policy is data; this application never sends refunds or changes an external system.

This differs from the existing `applications/workflow` executor: that application already owns durable effects and live order transitions. This application investigates several interacting systems after partial failures and proposes a proof-carrying reconciliation. It reuses the existing Folder, named-call, `nl`, `iterateOn`, service and verifier mechanisms. It introduces no new workflow engine, durable store or recovery runtime.

The application is natural-language policy throughout. Its `.nl` files spell out structured pseudocode with typed input/output, numbered loops, branches, state updates and evidence obligations. TypeScript is restricted to declarations, immutable source loading, exact arithmetic and independent verification of the produced artifacts.

## Observable result

`ForensicReport` contains:

- `identityDecisions`: request attempts grouped by an explicitly supported idempotency domain and payload meaning; incompatible payload reuse remains separate or conflicted.
- `standingEffects`: effect ID, business identity, quantity, observed/reversed/unknown status, and quotation witnesses.
- `causalEdges`: directed intent/receipt/compensation links; wall-clock similarity alone is not causality.
- `repairPolicy`: immediate actions and observation-dependent branches, with every branch's postconditions and irreversibility constraints.
- `unresolved`: exact propositions not established by the snapshot, and the smallest source query that would decide each relevant branch.
- `proof`: source citations, arithmetic reconciliation, coverage, and rejected tempting alternatives.

The output contract supplies identifiers, category vocabularies, order semantics and null conventions independently of any gold answer. Dynamic result maps use generic types; expected membership never appears in signatures.

## Callable module graph

The following names describe responsibilities, not a uniform pipeline. Every child physically belongs to its caller's companion folder. A root may invoke independent sibling branches, but a child never assumes access to its sibling.

```text
investigate/adjudicate/advanceLedger/assessEvidence.nl
investigate/adjudicate/advanceLedger.nl
investigate/adjudicate.nl
investigate/audit/challengePolicy.nl
investigate/audit.nl
investigate/correlate/alignTime/interpretClock.nl
investigate/correlate/alignTime.nl
investigate/correlate/bindAttempts/compareRequestMeaning/resolveKeyScope.nl
investigate/correlate/bindAttempts/compareRequestMeaning.nl
investigate/correlate/bindAttempts.nl
investigate/correlate/deriveCausality.nl
investigate/correlate.nl
investigate/inventory/interpretDocument.nl
investigate/inventory.nl
investigate/repair/chooseObservation/partitionAnswers.nl
investigate/repair/chooseObservation.nl
investigate/repair/deriveObligations.nl
investigate/repair/searchPolicy/expandFrontier/assessAction/interpretCompensationScope.nl
investigate/repair/searchPolicy/expandFrontier/assessAction.nl
investigate/repair/searchPolicy/expandFrontier.nl
investigate/repair/searchPolicy.nl
investigate/repair.nl
investigate.nl
```

First implemented slice: 23 `.nl` functions, 535 source lines including frontmatter, plus typed declarations. The earlier 30-function/650–900-line estimate was an architecture estimate; several proposed helpers were merged into their owning semantic stage when they would only have forwarded state. The deepest repair branch has six active named calls plus a consumed local literal. Inventory reduces source collections; correlation joins independently owned time, identity and causal analyses. Live traces must still demonstrate these authored contributions.

## Typed state and ownership

| Unit | Input → owned result | New semantic work |
|---|---|---|
| `inventory` | `Folder, InvestigationRequest → EvidenceIndex` | Partition documents by what they assert, retaining source IDs and exact spans. |
| Journal/receipt/contract interpreters | `SourceDocument, LocalContext → JournalFact[] / ReceiptFact[] / ContractClause[]` | Distinguish intention, observation, guarantee, attribution and exception scope. |
| `alignTime` | `ClockFacts, EventMentions → TimeConstraints` | Interpret local-clock corrections and logical-before statements without equating unrelated clocks. |
| `interpretClockClause` | `Clause, ClockContext → ClockRelation` | Resolve whether an offset applies before/after a restart or only to a named host. |
| `bindAttempts` | `Attempts, Clauses → IdentityPartition` | Join key namespace, tenant, migration epoch and request meaning. |
| `compareRequestMeaning` | `AttemptPair, RelevantClauses → PayloadRelation` | Decide semantic same request versus a different requested effect despite reused bytes or changed wording. |
| `resolveKeyScope` | `PayloadRelation, NamespaceHistory → ScopedIdentityDecision` | Apply the exact scope of deduplication and migration exceptions. |
| `deriveCausality` | `IdentityPartition, TimeConstraints, References → CausalGraph` | Add supported causality, retain concurrency and expose impossible reference cycles. |
| `adjudicate` | `EvidenceIndex, CausalGraph → EffectLedger` | Reduce the finite effect frontier, joining business identity with evidence strength. |
| `evaluateEvidence` | `EffectCandidate, Witnesses → EvidenceAssessment` | Separate acknowledgment, executed effect, delayed receipt, denial and missing evidence. |
| `reconcileEffect` | `EvidenceAssessment, PriorLedger → EffectDelta` | Decide standing, reversed, duplicate observation or unresolved effect. |
| `resolveConflict` | `ConflictingAssertions, SourceScopes → ConflictDecision` | Resolve only authority/scope differences supported by the supplied contract; otherwise preserve conflict. |
| `deriveObligations` | `EffectLedger, DesiredPostconditions → ObligationGraph` | Determine what is owed and which effects can no longer be undone. |
| `searchPolicy` | `ObligationGraph, AllowedActions → RepairSearchState` | Expand contingent repair candidates under dependency, quantity and irreversible-boundary constraints. |
| `assessCandidate` | `CandidatePolicy, Ledger, Clauses → CandidateAssessment` | Decide applicability and harmful side effects; rejection includes a minimal source-backed counterexample. |
| `chooseObservation` | `UnresolvedBranches, AvailableQueries → Discriminator` | Choose an observation whose possible answers actually separate required repairs. |
| `audit` | `ForensicReportDraft, Snapshot → AuditedReport` | Challenge each inference and validate every observation branch against the same evidence. |

All intermediate records carry `Witness {sourceId, path, quote}`. Unknown quantities and outcomes are null or explicit union arms; no fabricated placeholder fact satisfies a required string field. State changes are deltas consumed by the next stage, not explanatory strings beside an independently recomputed answer.

## Deep pseudocode and dynamic literals

1. `inventory` reduces nested source folders. Each folder is interpreted according to its source contract; the reducer preserves new assertions and unresolved source-role ambiguities.
2. `bindAttempts` iterates unresolved candidate joins. Inside `compareRequestMeaning`, create a typed NL literal scoped to the two requests and the selected identity clause. Its result must determine a merge/reject/defer delta that `bindAttempts` consumes.
3. `adjudicate` uses `iterateOn` over effect identities. The step invokes evidence-strength and replay judgments, applies one typed `EffectDelta`, and removes that identity from the pending set. A contradiction becomes an output; it does not create an unbounded retry.
4. `searchPolicy` maintains a finite frontier of action sets and unresolved observation branches. `expandFrontier` removes a candidate, generates only source-authorized extensions, and records rejected candidates. A dynamically formulated compensation-scope literal must determine which extension is legal. The structural measure is the number of not-yet-expanded members of the finite candidate universe, not a turn budget.
5. `chooseObservation` formulates a different local NL literal for each proposed query. It interprets the answers under the disputed proposition, partitions candidate worlds, and returns the partition actually consumed by the contingent plan.
6. `audit` follows each branch and asks `challengePolicy` to seek a source-backed counterexample. Exact verifiers check returned witnesses, accounting and state transitions; they do not choose a plan.

## Exact seams and why they are crisp

| Seam | Allowed exact work | Excluded policy |
|---|---|---|
| Folder and source snapshot | Read immutable files, preserve IDs/paths/content hashes. | Deciding what a statement means. |
| Numeric/time helpers | Add quantities and propagate already interpreted integer interval constraints. | Choosing the interpretation of a clock or contractual clause. |
| Witness verifier | Require cited path/ID correspondence and exact quote occurrence. | Deciding whether a quote supports a claim. |
| Policy verifier | Replay declared action deltas against the formal candidate world; check conservation, preconditions and all reported branches. | Inventing effect identities, deciding source authority, selecting the best repair. |
| Corpus packager | Preserve complete `.nl` graph and typed declarations; place oracle outside model-visible project files. | Generating target answers or flattening the application into a generic chain. |

No policy implementation in a TypeScript helper, hidden gold-backed service or hand-coded answer synthesizer is permitted. If a hot policy later needs a crisp implementation, it must use the existing pluggable interface and earn separate qualification.

## Concrete first world and quality gates

The first world contains 40 connected clauses in 16 nested source documents across two sagas sharing a reservation pool, tenant-scoped bank keys and warehouse epoch changes. A lost acknowledgment, an intentional retry, changed request meaning under one reused key, an observed compensation, a shipment boundary and an unresolved receipt must all alter the resulting policy. Each fact has a documented downstream consumer; irrelevant padding is excluded.

At least three source-preserving task variants will ask for reconstruction, repair and a discriminating observation. These are views of one world and share one split group. At least one counterfactual changes the correct business effect despite similar surface wording, and another changes the permissible repair without changing gross totals.

Before generation readiness: parse/type/callable-graph checks; independent manual source/oracle review; exact reference replay and negative verifier mutations; source-visibility check; then live execution of actual NL sources. The live review measures active-parent nesting, meaningful owned deltas, consumed dynamic literals and all branch outcomes. About 48 samples per adopted model-facing presentation remain required; one successful run is a diagnostic. Training admission is separate.

## Implemented-slice responsibility consolidation

- `interpretDocument` extracts witnessed atoms from all source roles; journal/receipt/contract differences remain semantic classifications, without three repetitive file wrappers.
- `advanceLedger` combines reconciliation and replay accounting after its `assessEvidence` child; conflicts stay explicit in ledger state.
- `deriveCausality` owns edge construction directly; there is no per-edge forwarding helper.
- `deriveObligations` owns the complete cancellation/price/custody join. Its result is consumed by observation planning and search.
- `expandFrontier` owns candidate construction; `assessAction` owns validity and calls `interpretCompensationScope` for the local rollback judgment.
- No TypeScript policy helper exists. `types.ts` contains declarations only. Any later external checker consumes results and evidence without synthesizing a repair.

Some earlier table names describe responsibilities now consolidated above rather than extra callable files. The file graph is the authoritative implemented graph. The first-world oracle is a canonical semantic witness; independent actions can be delayed into every observation branch when dependencies and final effects remain equivalent.

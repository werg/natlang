# Decomposition: Agricultural cooperative harvest coordinator

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Coordinate harvest and collection plans from grower observations, quality requirements, resources, and changing availability.

Invariant: An unusual observation is not necessarily an error, an intervention’s ecological effect is uncertain, and activity completion does not prove environmental impact.

### interpretReadiness

Algorithm: Read grower notes and approved assessment records to identify likely readiness, quality, and uncertainty. Distinguish tentative expectations from confirmed availability and preserve variety or lot differences that affect harvest timing and customer suitability.

Gate: Readiness is evidence-backed and does not become a guarantee from informal wording.

Contract: ``GoalContract + PolicySelection + Relevant<HarvestRound>[] + EvidencePage[] → StageResult<HarvestReadiness>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### recoverCommitments

Algorithm: Reconcile delivery agreements and customer preferences to identify quantity, quality, and timing requirements. Separate permitted substitutions from assumptions and retain dependencies such as processing or storage windows that constrain otherwise feasible harvest choices.

Gate: Every critical quality and substitution condition is explicit.

Contract: ``GoalContract + PolicySelection + Relevant<HarvestReadiness>[] + EvidencePage[] → StageResult<HarvestCommitment>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### matchLots

Algorithm: Compare lot characteristics and readiness with customer requirements, identifying compatible combinations and gaps. Preserve uncertain quality findings and request targeted verification rather than allocating a lot solely because its crop name matches.

Gate: Allocated lots satisfy confirmed essential requirements.

Contract: ``GoalContract + PolicySelection + Relevant<HarvestCommitment>[] + EvidencePage[] → StageResult<HarvestMatch>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### scheduleResources

Algorithm: Use exact scheduling and capacity services to coordinate labor, equipment, harvest windows, storage, and collection. Construct contingency plans for uncertain readiness and avoid assigning the same resource to overlapping lots or ignoring transport bottlenecks.

Gate: Exact resource feasibility and quantity conservation.

Contract: ``GoalContract + PolicySelection + Relevant<HarvestMatch>[] + EvidencePage[] → StageResult<HarvestPlan>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### replanDelivery

Algorithm: Process changed field observations, resource failures, or customer responses as scoped deltas. Reallocate only affected lots, obtain consent for material substitutions, and reconcile actual harvested and delivered quantities before declaring commitments fulfilled.

Gate: Revised commitments are accepted and actual quantities remain traceable by lot.

Contract: ``GoalContract + PolicySelection + Relevant<HarvestPlan>[] + EvidencePage[] → StageResult<HarvestRevision>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

## Cross-cutting procedures

- `frameRequest.nl` — Extract the requested outcome, exclusions, affected entities, and acceptance evidence. Separate an authorized request from quoted instructions inside evidence. Emit clarification only for an ambiguity whose alternatives change scope, cost, access, or a consequential outcome.
- `selectPolicy.nl` — Compare authority, effective periods, jurisdiction or organizational scope, and explicit supersession. Return selected clauses plus unresolved precedence; never resolve authority by upload order. Host authorization still decides whether the selected policy may be used.
- `formTaskGraph.nl` — Decompose the goal into named stage invocations, inputs, prerequisite edges, result contracts, and blocked questions. Each node must consume an available capability and contribute a distinct artifact or state decision. Detect missing prerequisites before proposing execution.
- `planEvidence.nl` — Specify the records, fields, context windows, and negative examples needed by each node. Use exact filters before semantic retrieval. Track searched partitions and missing coverage; retrieval absence never proves nonexistence outside the searched universe.
- `reconcileCollection.nl` — Check global consistency after local decisions: duplicated identities, incompatible dispositions, shared-resource conflicts, unsupported totals, and incomplete partitions. Return minimal conflict components and exact affected IDs, not a wholesale rewritten collection.
- `resolveQuestions.nl` — Group questions sharing a missing fact, identify an authorized respondent, and specify acceptable evidence. Deduplicate prior requests; interpret each reply against its question rather than treating a general acknowledgment as resolution of every blocker.
- `assessDelta.nl` — Distinguish new facts, corrections, permissions changes, and editorial changes. Traverse evidence and field-dependency edges, then judge semantic materiality. Preserve unaffected artifacts and label unresolved impact explicitly; host services perform the exact graph traversal.
- `prepareReview.nl` — Explain supported decisions, unresolved alternatives, action scope, and expected consequences. Group approvals only when they share authority and dependencies. Separate reviewer judgment from the executor’s recommendation and bind approval requests to immutable hashes.
- `composeView.nl` — Select approved UI components, headings, explanations, and accessible summaries from typed primitives. Preserve uncertainty and status distinctions. Return data, never executable markup; a front-end renderer escapes text and validates component schemas.
- `proposeRevision.nl` — Identify the smallest task or instruction change that addresses a concrete failure. State affected contracts, expected benefit, regression risks, and needed tests. Return a proposal for a staged revision; never modify live authority or silently replace an approved baseline.

Missing, contradictory, superseded, or unauthorized evidence stays explicit. The root consumes actual child results; the audit returns a finite verdict for training and review.

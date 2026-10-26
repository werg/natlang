# Decomposition: Community garden steward

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Coordinate planting concepts, shared resources, member agreements, and seasonal work while preserving consent and practical constraints.

Invariant: An unusual observation is not necessarily an error, an intervention’s ecological effect is uncertain, and activity completion does not prove environmental impact.

### interpretPreferences

Algorithm: Read member requests and prior agreements to identify desired uses, availability, and explicitly stated access needs. Separate preferences from accepted communal obligations and avoid inferring personal circumstances or assigning work without consent.

Gate: Member commitments are explicit and attributable.

Contract: ``GoalContract + PolicySelection + Relevant<GardenSeason>[] + EvidencePage[] → StageResult<GardenPreference>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### modelPlots

Algorithm: Reconcile plot records, field observations, and supplied gardening guidance to describe known conditions and constraints. Preserve uncertainty about soil or other untested properties, and route consequential interventions to appropriate local expertise.

Gate: Planning uses verified or explicitly uncertain conditions, not invented agronomic facts.

Contract: ``GoalContract + PolicySelection + Relevant<GardenPreference>[] + EvidencePage[] → StageResult<PlotConstraint>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### generatePlans

Algorithm: Propose varied planting and shared-space concepts consistent with supplied guidance and member goals. Include creative communal uses, preserve access paths and resource limits, and identify tradeoffs rather than optimizing only one member’s preferences.

Gate: Known hard constraints and approved garden rules remain satisfied.

Contract: ``GoalContract + PolicySelection + Relevant<PlotConstraint>[] + EvidencePage[] → StageResult<GardenPlan>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### negotiateAllocation

Algorithm: Apply supplied allocation principles to feasible plans and propose fair compromises for contested plots or duties. Distinguish a suggestion from accepted responsibility and use reversible trials where preferences conflict without a clear rule-based resolution.

Gate: No unconsented duty or opaque allocation criterion.

Contract: ``GoalContract + PolicySelection + Relevant<GardenPlan>[] + EvidencePage[] → StageResult<GardenAgreement>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### coordinateSeason

Algorithm: Turn accepted plans into bounded seasonal tasks, shared-tool reservations, and reminders. Adapt to changed availability or observations, preserve agreements across seasons, and record actual completion without treating missed tasks as automatic grounds for exclusion.

Gate: Resource scheduling and member consent remain current through replanning.

Contract: ``GoalContract + PolicySelection + Relevant<GardenAgreement>[] + EvidencePage[] → StageResult<GardenTask>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

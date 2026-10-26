# Decomposition: Facility complaint-to-work-order system

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Interpret occupant reports, coordinate the correct trades, and verify that completed work resolves the reported problem.

Invariant: A site instruction may not authorize a commercial change, similar bid totals may cover different scope, and a complete dossier does not prove the building is technically safe.

### interpretComplaint

Algorithm: Read occupant descriptions to identify location, symptom, timing, and practical impact. Distinguish observed behavior from guessed causes, resolve ambiguous room names, and preserve access or communication preferences without inferring sensitive personal circumstances.

Gate: Location and symptom are established or a targeted clarification is requested.

Contract: ``GoalContract + PolicySelection + Relevant<FacilityCase>[] + EvidencePage[] → StageResult<FacilityObservation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### linkRelatedIssues

Algorithm: Compare reports by location, asset dependencies, timing, and symptom mechanism. Group likely shared issues for coordinated investigation while retaining unrelated complaints that use similar language; do not close a new report solely because a nearby task was completed.

Gate: Grouping has evidence and preserves independent outcomes.

Contract: ``GoalContract + PolicySelection + Relevant<FacilityObservation>[] + EvidencePage[] → StageResult<FacilityIssueGroup>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### formWorkOrders

Algorithm: Choose approved inspection or repair task types and required trade qualifications. Sequence dependencies, include access arrangements and occupant impact, and route safety-sensitive or uncertain cases to qualified personnel rather than generating improvised repairs.

Gate: Task scope and qualification match the observed issue and approved procedure.

Contract: ``GoalContract + PolicySelection + Relevant<FacilityIssueGroup>[] + EvidencePage[] → StageResult<FacilityWorkOrder>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### coordinateVisits

Algorithm: Schedule visits using exact availability and resource constraints, obtain required access consent, and coordinate multiple trades. Preserve cancellations and failed-access events, and update occupants with truthful windows rather than unsupported precise arrival promises.

Gate: No unauthorized access or impossible trade sequence.

Contract: ``GoalContract + PolicySelection + Relevant<FacilityWorkOrder>[] + EvidencePage[] → StageResult<FacilityVisit>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### verifyResolution

Algorithm: Compare technician completion evidence and subsequent occupant observations with the original complaint. Distinguish a replaced component from resolution of the symptom, detect recurrence, and reopen only the relevant issue when work does not achieve the intended outcome.

Gate: Closure requires the defined outcome evidence or an explicit accepted monitoring state.

Contract: ``GoalContract + PolicySelection + Relevant<FacilityVisit>[] + EvidencePage[] → StageResult<FacilityResolution>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

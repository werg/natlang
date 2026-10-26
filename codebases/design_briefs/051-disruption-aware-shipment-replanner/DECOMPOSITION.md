# Decomposition: Disruption-aware shipment replanner

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Recover shipping plans after disruptions while preserving critical commitments, exact capacity, and truthful customer communications.

Invariant: An item description is not verified compatibility, a reservation is not a transfer, and a creative match must preserve essential recipient and handling constraints.

### interpretDisruption

Algorithm: Read carrier notices and operational reports to determine what is unavailable, for how long, and with what certainty. Distinguish a delayed service from a closed route and preserve exceptions that may allow some shipment classes to continue.

Gate: Scope is evidence-backed and uncertain reopening times are not treated as confirmed capacity.

Contract: ``GoalContract + PolicySelection + Relevant<ShipmentDisruption>[] + EvidencePage[] → StageResult<DisruptionImpact>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### recoverPromises

Algorithm: Reconcile order terms, customer messages, and product constraints to identify what must be preserved. Separate contractual deadlines from preferences, recognize dependencies such as installation appointments, and avoid ranking customers by unstated commercial assumptions.

Gate: Every priority follows the supplied policy and actual commitment evidence.

Contract: ``GoalContract + PolicySelection + Relevant<DisruptionImpact>[] + EvidencePage[] → StageResult<ShippingCommitment>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### constructAlternatives

Algorithm: Generate permitted rerouting, split-shipment, consolidation, or substitution options. Explain which commitments each option preserves or changes, and exclude alternatives violating handling requirements or relying on unverified carrier capacity.

Gate: Essential constraints remain intact; changed promises require explicit customer or owner approval.

Contract: ``GoalContract + PolicySelection + Relevant<ShippingCommitment>[] + EvidencePage[] → StageResult<ShippingAlternative>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### allocateCapacity

Algorithm: Submit candidate plans to exact capacity and timing services, then compare feasible allocations under supplied priorities. Reserve selected capacity before promising it, retain unfulfilled demand explicitly, and detect conflicts across shipments sharing scarce resources.

Gate: Capacity conservation, route feasibility, and exact cost limits pass.

Contract: ``GoalContract + PolicySelection + Relevant<ShippingAlternative>[] + EvidencePage[] → StageResult<ShippingAllocation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### coordinateRecovery

Algorithm: Sequence approved bookings and communications, reconcile carrier acknowledgments, and track actual shipment progress against revised promises. Replan only affected legs when new events arrive and explain residual uncertainty rather than claiming recovery after booking alone.

Gate: Actual carrier state and customer-approved commitments agree before closure.

Contract: ``GoalContract + PolicySelection + Relevant<ShippingAllocation>[] + EvidencePage[] → StageResult<ShippingRecovery>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

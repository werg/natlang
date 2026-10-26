# Decomposition: Industrial surplus exchange

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Discover useful reuse matches and coordinate multi-party transfers while preserving compatibility, ownership, and allocation constraints.

Invariant: An item description is not verified compatibility, a reservation is not a transfer, and a creative match must preserve essential recipient and handling constraints.

### interpretOffers

Algorithm: Translate surplus descriptions into functional properties, quantities, condition, and known limitations. Distinguish verified specifications from seller assumptions, preserve ownership and handling restrictions, and identify information needed before proposing a consequential reuse.

Gate: No unsupported material property or transferable-title claim.

Contract: ``GoalContract + PolicySelection + Relevant<SurplusExchange>[] + EvidencePage[] → StageResult<SurplusOffer>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### interpretNeeds

Algorithm: Recover the actual purpose behind recipient requests, separating essential performance from preferred form. Identify acceptable adaptations and nonnegotiable constraints, including evidence or specialist approval needed for unfamiliar materials or equipment.

Gate: Recipient confirms constraints that determine suitability or safety.

Contract: ``GoalContract + PolicySelection + Relevant<SurplusOffer>[] + EvidencePage[] → StageResult<ReuseNeed>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### generateReuse

Algorithm: Propose conventional and inventive matches based on function rather than product names. Explain the adaptation mechanism and uncertainty, include multi-item bundles when useful, and reject creativity that depends on relaxing essential requirements or unapproved handling.

Gate: Essential fit is evidenced; speculative compatibility creates a qualification task, not an allocation.

Contract: ``GoalContract + PolicySelection + Relevant<ReuseNeed>[] + EvidencePage[] → StageResult<ReuseMatch>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### assembleExchange

Algorithm: Construct feasible multi-party exchanges using exact quantity, reservation, and transport services. Account for dependent legs, storage windows, and adaptation costs, and preserve unallocated surplus and unmet needs explicitly.

Gate: No double allocation, unsupported capacity, or negative quantity.

Contract: ``GoalContract + PolicySelection + Relevant<ReuseMatch>[] + EvidencePage[] → StageResult<ExchangePackage>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### verifyTransfer

Algorithm: Coordinate approved transfers and verify quantity, condition, ownership, and recipient acceptance at handoff. Record rejected or partially accepted material and update future matching evidence without falsely reporting a completed reuse benefit.

Gate: Transfer and acceptance evidence support reported quantities and outcomes.

Contract: ``GoalContract + PolicySelection + Relevant<ExchangePackage>[] + EvidencePage[] → StageResult<ReuseReceipt>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

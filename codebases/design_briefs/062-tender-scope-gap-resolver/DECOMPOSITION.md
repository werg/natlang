# Decomposition: Tender scope gap resolver

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Reconcile bids with differing inclusions and exclusions into a complete, accepted responsibility matrix without hidden gaps or duplication.

Invariant: A site instruction may not authorize a commercial change, similar bid totals may cover different scope, and a complete dossier does not prove the building is technically safe.

### normalizeBids

Algorithm: Read each bid, qualification, schedule, and addendum as a package. Extract functional scope rather than relying on trade headings, distinguish allowances from firm inclusions, and retain conditional exclusions that affect interfaces with other bidders.

Gate: Every scope statement retains its original qualification and bid version.

Contract: ``GoalContract + PolicySelection + Relevant<ScopeReview>[] + EvidencePage[] → StageResult<BidScope>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### mapRequirements

Algorithm: Compare required functions with bid inclusions and exclusions, recognizing different descriptions of the same work. Separate supply, installation, commissioning, and certification responsibilities instead of assuming one broad line covers all phases.

Gate: Coverage is phase-specific and evidence-backed.

Contract: ``GoalContract + PolicySelection + Relevant<BidScope>[] + EvidencePage[] → StageResult<ScopeCoverage>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### identifySeams

Algorithm: Inspect interfaces where one bidder excludes work as by others or assumes another trade provides prerequisites. Identify unowned functions and duplicated commitments, and distinguish acceptable overlap from double-charged or conflicting responsibility.

Gate: Every required interface has an accepted owner or an explicit unresolved gap.

Contract: ``GoalContract + PolicySelection + Relevant<ScopeCoverage>[] + EvidencePage[] → StageResult<ScopeInterface>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### runClarifications

Algorithm: Form precise clarification requests describing the disputed boundary and desired acceptance. Interpret replies at the commitment level, preserve qualifications, and avoid treating general cooperation language as acceptance of a specific responsibility.

Gate: A clarification closes a gap only with explicit scoped acceptance.

Contract: ``GoalContract + PolicySelection + Relevant<ScopeInterface>[] + EvidencePage[] → StageResult<ScopeClarification>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### freezeMatrix

Algorithm: Assemble a reconciled responsibility matrix and commercial comparison using accepted clarifications. Recheck all required functions and dependencies, expose residual gaps, and bind the selection packet to the exact bid editions reviewed.

Gate: No silent assignment of unaccepted obligations and no uncovered mandatory function.

Contract: ``GoalContract + PolicySelection + Relevant<ScopeClarification>[] + EvidencePage[] → StageResult<ResponsibilityMatrix>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

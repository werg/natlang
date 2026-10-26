# Decomposition: Repair cooperative dispatcher

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Turn repair requests into bounded diagnostics, qualified assignments, and verified outcomes while learning from recurring cases.

Invariant: An item description is not verified compatibility, a reservation is not a transfer, and a creative match must preserve essential recipient and handling constraints.

### interpretSymptoms

Algorithm: Read the request and prior attempts to separate observations from the requester’s diagnosis. Identify approved handling restrictions and information needed for safe routing, without inventing technical causes or giving unqualified hazardous repair instructions.

Gate: Observations and inferred causes remain distinct; risk flags trigger specialist routing.

Contract: ``GoalContract + PolicySelection + Relevant<RepairRequest>[] + EvidencePage[] → StageResult<RepairSymptom>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### planDiagnostics

Algorithm: Choose a bounded diagnostic task from supplied safe procedures that distinguishes plausible issues. Specify required qualifications, tools, and stop conditions; route work beyond cooperative capability to an appropriate specialist rather than improvising.

Gate: Diagnostic is approved, informative, and within assigned capability.

Contract: ``GoalContract + PolicySelection + Relevant<RepairSymptom>[] + EvidencePage[] → StageResult<RepairDiagnostic>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### matchRepairers

Algorithm: Match tasks to volunteers or specialists using stated skills, verified qualifications where required, availability, and consent. Do not infer capability from identity or enthusiasm, and preserve supervision requirements for trainees.

Gate: Exact scheduling and required qualification checks pass.

Contract: ``GoalContract + PolicySelection + Relevant<RepairDiagnostic>[] + EvidencePage[] → StageResult<RepairAssignment>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### coordinateWork

Algorithm: Sequence inspection, parts procurement, repair, and verification under approved procedures. Reserve shared tools, obtain consent for cost or scope changes, and preserve an explicit stop-and-escalate path when findings exceed the original plan.

Gate: No unauthorized repair scope, double-booked resource, or unapproved part substitution.

Contract: ``GoalContract + PolicySelection + Relevant<RepairAssignment>[] + EvidencePage[] → StageResult<RepairWorkPlan>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### captureOutcomes

Algorithm: Compare post-work checks and customer observations with the original symptoms. Record partial success, recurrence, and specialist referrals, then extract scoped lessons that improve future routing without treating one repair as universal proof of a cause.

Gate: Outcome is verified and lessons retain applicability limits.

Contract: ``GoalContract + PolicySelection + Relevant<RepairWorkPlan>[] + EvidencePage[] → StageResult<RepairOutcome>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

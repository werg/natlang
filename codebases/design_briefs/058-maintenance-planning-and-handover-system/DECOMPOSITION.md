# Decomposition: Maintenance planning and handover system

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Convert approved manuals and service observations into executable maintenance plans and accurate shift handovers.

Invariant: A plausible cause is not a verified cause, local exceptions do not remove safety prerequisites, and documentary completeness does not certify physical safety.

### interpretObservations

Algorithm: Reconcile inspection notes, service history, and operator reports to identify observed conditions and unresolved discrepancies. Separate symptoms from diagnosis, retain intermittent behavior, and identify information needed for routing under approved maintenance procedures.

Gate: Observations do not become established faults without supporting evidence.

Contract: ``GoalContract + PolicySelection + Relevant<MaintenanceCase>[] + EvidencePage[] → StageResult<AssetObservation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### selectProcedure

Algorithm: Match the asset variant and observed condition to approved procedures, checking revision, qualification, and required isolation or access conditions. Escalate unsupported cases rather than inventing maintenance instructions from general knowledge.

Gate: Correct asset and procedure edition with all critical prerequisites established.

Contract: ``GoalContract + PolicySelection + Relevant<AssetObservation>[] + EvidencePage[] → StageResult<MaintenanceProcedureFit>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### constructWorkPlan

Algorithm: Decompose the approved procedure into accountable tasks, resource reservations, and verification steps. Sequence shared access and parts availability, preserve stop conditions, and distinguish planned work from authorized readiness to begin.

Gate: Exact resource feasibility and no bypass of required safety or qualification checks.

Contract: ``GoalContract + PolicySelection + Relevant<MaintenanceProcedureFit>[] + EvidencePage[] → StageResult<MaintenanceWorkOrder>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### interpretProgress

Algorithm: Compare technician notes and system events with task completion criteria. Distinguish work attempted, work completed, and verification passed; preserve changed conditions and unexpected findings that alter the remaining plan.

Gate: No completion based solely on free-text optimism or elapsed time.

Contract: ``GoalContract + PolicySelection + Relevant<MaintenanceWorkOrder>[] + EvidencePage[] → StageResult<MaintenanceProgress>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### composeHandover

Algorithm: Prepare a shift handover describing actual asset state, work in progress, unresolved findings, and required next checks. Require acknowledgment from the receiving owner and preserve the distinction between maintenance completion and authorized return to service.

Gate: Handover includes all open obligations and current safety-relevant state.

Contract: ``GoalContract + PolicySelection + Relevant<MaintenanceProgress>[] + EvidencePage[] → StageResult<MaintenanceHandover>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

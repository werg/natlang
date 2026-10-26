# Decomposition: Incident recovery orchestrator

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Maintain evidence-based hypotheses, choose approved diagnostics, and coordinate verified service recovery without uncontrolled production actions.

Invariant: Passing a test does not prove the requirement was understood, type compatibility does not guarantee behavioral compatibility, and a plausible incident story is not a verified cause.

### frameImpact

Algorithm: Combine alerts, user reports, and service telemetry to describe observed impact and uncertainty. Separate symptoms from causes and distinguish missing telemetry from healthy behavior; identify the service outcomes that recovery must actually restore.

Gate: Impact statements cite observations and preserve unknown coverage.

Contract: ``GoalContract + PolicySelection + Relevant<Incident>[] + EvidencePage[] → StageResult<IncidentImpact>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### maintainHypotheses

Algorithm: Generate a bounded set of causal hypotheses from deployment history, dependencies, and symptoms. Track evidence for and against each, choose checks that distinguish mechanisms, and avoid repeatedly confirming the most recent or most fluent explanation.

Gate: Hypotheses remain provisional until diagnostic evidence supports them.

Contract: ``GoalContract + PolicySelection + Relevant<IncidentImpact>[] + EvidencePage[] → StageResult<IncidentHypothesis>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### selectDiagnostics

Algorithm: Choose from approved read-only or bounded diagnostic operations, considering expected discrimination and operational risk. Validate scope and rate limits, request commander approval for intrusive checks, and record what each possible result would change.

Gate: Diagnostic action is allowlisted and cannot widen credentials or query scope.

Contract: ``GoalContract + PolicySelection + Relevant<IncidentHypothesis>[] + EvidencePage[] → StageResult<DiagnosticAction>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### planRecovery

Algorithm: Select a recovery path from approved runbooks using current evidence and legal action options. Sequence reversible actions first where appropriate, preserve prerequisites, and avoid improvising destructive commands or treating a successful command as recovered service.

Gate: Current authority, service state, and runbook preconditions checked immediately before execution.

Contract: ``GoalContract + PolicySelection + Relevant<DiagnosticAction>[] + EvidencePage[] → StageResult<RecoveryStep>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### verifyRestoration

Algorithm: Recheck the affected user-visible functions and representative cohorts after recovery actions. Distinguish reduced alerts from actual restoration, retain residual degradation, and schedule observation long enough to detect recurrence under the supplied incident policy.

Gate: Closure requires outcome verification, not merely green infrastructure metrics.

Contract: ``GoalContract + PolicySelection + Relevant<RecoveryStep>[] + EvidencePage[] → StageResult<RestorationFinding>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

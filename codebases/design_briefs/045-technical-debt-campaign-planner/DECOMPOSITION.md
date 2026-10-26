# Decomposition: Technical debt campaign planner

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Turn recurring engineering friction into bounded improvement programs with causal hypotheses, dependencies, and verifiable completion.

Invariant: Passing a test does not prove the requirement was understood, type compatibility does not guarantee behavioral compatibility, and a plausible incident story is not a verified cause.

### collectSymptoms

Algorithm: Process developer reports, incidents, and issue histories to identify recurring friction. Separate specific observed costs from broad complaints, deduplicate repeated descriptions of one event, and preserve minority workflows that may face severe but infrequent problems.

Gate: Symptom frequency and impact are grounded in observable evidence.

Contract: ``GoalContract + PolicySelection + Relevant<DebtCampaign>[] + EvidencePage[] → StageResult<DebtSymptom>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### formMechanisms

Algorithm: Group symptoms by plausible shared mechanisms such as unstable interfaces or manual coordination. Distinguish architectural causes from staffing or process issues, and formulate checks that could disconfirm the proposed technical explanation.

Gate: A popular explanation is not treated as established causality.

Contract: ``GoalContract + PolicySelection + Relevant<DebtSymptom>[] + EvidencePage[] → StageResult<DebtMechanism>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### designWorkPackages

Algorithm: Convert supported mechanisms into finite improvement packages with explicit interfaces and non-goals. Propose alternatives with different disruption levels, avoid unbounded rewrites, and define completion through changed behavior rather than merged code alone.

Gate: Each package is executable and tied to the original symptom or validated mechanism.

Contract: ``GoalContract + PolicySelection + Relevant<DebtMechanism>[] + EvidencePage[] → StageResult<DebtWorkPackage>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### sequenceCampaign

Algorithm: Order enabling changes, migrations, and cleanup using exact dependency and capacity services. Preserve ongoing delivery constraints, identify irreversible steps, and create milestones that deliver useful improvements before the entire campaign is complete.

Gate: Feasible capacity, acyclic dependencies, and safe migration sequencing.

Contract: ``GoalContract + PolicySelection + Relevant<DebtWorkPackage>[] + EvidencePage[] → StageResult<DebtWave>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### verifyImprovement

Algorithm: Compare post-change workflow evidence with the original problem definition, accounting for changes in workload and observation. Distinguish successful delivery from actual improvement and reopen mechanisms that the completed work did not address.

Gate: Completion requires evidence of the intended operational effect, not only task closure.

Contract: ``GoalContract + PolicySelection + Relevant<DebtWave>[] + EvidencePage[] → StageResult<DebtOutcome>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

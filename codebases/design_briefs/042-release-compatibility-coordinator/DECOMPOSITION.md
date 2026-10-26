# Decomposition: Release compatibility coordinator

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Detect semantic compatibility risks and coordinate migrations across integrations, documentation, customer configurations, and operations.

Invariant: Passing a test does not prove the requirement was understood, type compatibility does not guarantee behavioral compatibility, and a plausible incident story is not a verified cause.

### interpretChanges

Algorithm: Read release diffs, requirements, and observed behavior to identify changed semantics, defaults, timing, and error handling. Distinguish internal refactoring from externally meaningful changes even when schemas and type signatures remain compatible.

Gate: Every claimed behavioral change has evidence or a targeted verification task.

Contract: ``GoalContract + PolicySelection + Relevant<ReleaseReview>[] + EvidencePage[] → StageResult<BehavioralChange>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### mapConsumers

Algorithm: Relate integrations, customer configurations, examples, and runbooks to the changed behavior. Identify implicit reliance such as ordering or retry assumptions, and preserve uncertain dependencies for targeted replay rather than declaring them unaffected.

Gate: Dependency claims are specific to behavior, not mere package-name overlap.

Contract: ``GoalContract + PolicySelection + Relevant<BehavioralChange>[] + EvidencePage[] → StageResult<ConsumerDependency>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### replayCompatibility

Algorithm: Execute representative consumer scenarios against baseline and candidate in isolated environments. Compare intended outcomes rather than raw output equality alone, and identify whether a difference is acceptable, breaking, or dependent on an unresolved contract.

Gate: Paired replay uses identical inputs and independent consumer acceptance criteria.

Contract: ``GoalContract + PolicySelection + Relevant<ConsumerDependency>[] + EvidencePage[] → StageResult<CompatibilityTest>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### planMigration

Algorithm: Construct staged migration plans with prerequisites, owners, compatibility shims where approved, and rollback conditions. Group consumers by meaningful risk and avoid advancing a wave before its required operational readiness evidence exists.

Gate: Each wave is feasible and has a tested rollback or explicit irreversible-change approval.

Contract: ``GoalContract + PolicySelection + Relevant<CompatibilityTest>[] + EvidencePage[] → StageResult<MigrationWave>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### coordinateRelease

Algorithm: Reconcile code, documentation, support guidance, and customer notices into one release state. Explain audience-specific consequences and verify that approved migrations and communications match the exact release artifact being promoted.

Gate: No release readiness claim over mismatched versions or unresolved blocking compatibility failures.

Contract: ``GoalContract + PolicySelection + Relevant<MigrationWave>[] + EvidencePage[] → StageResult<ReleaseReadiness>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

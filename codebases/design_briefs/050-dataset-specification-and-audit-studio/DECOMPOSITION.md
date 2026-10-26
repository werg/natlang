# Decomposition: Dataset specification and audit studio

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Translate intended dataset use into acquisition, annotation, coverage, and acceptance requirements with independently auditable gaps.

Invariant: A matching label is not semantic equivalence, a missing value may carry meaning, and exact aggregation is correct only after the intended population and predicates are established.

### defineUseContract

Algorithm: Interpret the intended application and decisions the dataset will support. Define target population, task boundaries, prohibited uses, and meaningful evaluation slices; avoid treating easily available data as representative of the intended use.

Gate: Every required slice and restriction has a use-based rationale.

Contract: ``GoalContract + PolicySelection + Relevant<DatasetProject>[] + EvidencePage[] → StageResult<DatasetUse>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### writeAnnotationSpec

Algorithm: Draft operational labeling instructions with examples, counterexamples, and uncertainty handling. Separate observable labels from inferred attributes, define adjudication, and test whether independent annotators can apply the rule consistently.

Gate: Labels are grounded in available evidence and ambiguous cases have an explicit disposition.

Contract: ``GoalContract + PolicySelection + Relevant<DatasetUse>[] + EvidencePage[] → StageResult<AnnotationRule>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### auditCandidates

Algorithm: Inspect candidate records for task relevance, data quality, provenance, consent, and leakage risks. Distinguish missing coverage from invalid records, and quarantine material whose rights or source integrity cannot be established.

Gate: No release candidate without verified provenance and permitted intended use.

Contract: ``GoalContract + PolicySelection + Relevant<AnnotationRule>[] + EvidencePage[] → StageResult<DatasetRecordAudit>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### measureCoverage

Algorithm: Compute exact slice coverage and overlap using the approved population definitions. Interpret whether apparent balance masks duplicated sources or missing mechanisms, and identify the most consequential acquisition gaps rather than optimizing superficial counts.

Gate: Exact counts, independent-source accounting, and no false representativeness claim.

Contract: ``GoalContract + PolicySelection + Relevant<DatasetRecordAudit>[] + EvidencePage[] → StageResult<CoverageFinding>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### commissionAndRelease

Algorithm: Create targeted acquisition or annotation tasks, rerun audits on new material, and assemble a release manifest with known limits. Freeze splits by source and world family, prevent leakage, and require owner approval for the specific release use.

Gate: Rights, quality, split integrity, and required coverage gates pass or limitations block release.

Contract: ``GoalContract + PolicySelection + Relevant<CoverageFinding>[] + EvidencePage[] → StageResult<DatasetRelease>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

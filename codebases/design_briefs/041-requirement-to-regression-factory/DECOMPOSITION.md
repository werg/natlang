# Decomposition: Requirement-to-regression factory

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Turn requirements and incident histories into independently validated regression tests that expose real behavioral defects.

Invariant: Passing a test does not prove the requirement was understood, type compatibility does not guarantee behavioral compatibility, and a plausible incident story is not a verified cause.

### recoverBehavior

Algorithm: Interpret prose requirements, bug reports, and relevant discussion to identify observable behavior, preconditions, and exclusions. Separate implementation suggestions from requirements and preserve ambiguous expectations for owner resolution before encoding them as tests.

Gate: Every assertion traces to a requirement or explicitly approved interpretation.

Contract: ``GoalContract + PolicySelection + Relevant<RegressionProject>[] + EvidencePage[] → StageResult<BehavioralRequirement>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### modelPartitions

Algorithm: Partition input and state space by behaviorally meaningful distinctions, including invalid inputs and interacting requirements. Identify boundary cases and previously untested combinations rather than generating many superficial variants of the same happy path.

Gate: Partitions cover each requirement’s decisive conditions and failure behavior.

Contract: ``GoalContract + PolicySelection + Relevant<BehavioralRequirement>[] + EvidencePage[] → StageResult<TestPartition>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### constructTests

Algorithm: Create executable test specifications using approved harness capabilities and isolated fixtures. Define deterministic setup, observable assertions, and cleanup; avoid brittle checks on incidental wording or timing unless those are actually contractual.

Gate: Tests compile and have independent expected behavior, not output copied from the implementation.

Contract: ``GoalContract + PolicySelection + Relevant<TestPartition>[] + EvidencePage[] → StageResult<RegressionTest>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### interpretFailures

Algorithm: Compare failures with requirements, execution traces, and fixture health. Distinguish a product defect from a bad oracle, flaky setup, or unavailable dependency; propose the smallest repair without weakening a valid assertion to match defective behavior.

Gate: Failure classification is evidence-backed and environment errors are not counted as product coverage.

Contract: ``GoalContract + PolicySelection + Relevant<RegressionTest>[] + EvidencePage[] → StageResult<TestFailureFinding>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### qualifySuite

Algorithm: Run independent seeded defects and mutation tests to evaluate whether the suite detects meaningful violations. Remove redundant tests only when coverage remains, preserve regression cases for repaired bugs, and report uncovered requirements explicitly.

Gate: Suite catches required seeded defects and does not pass merely because all tests match current behavior.

Contract: ``GoalContract + PolicySelection + Relevant<TestFailureFinding>[] + EvidencePage[] → StageResult<SuiteQualification>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

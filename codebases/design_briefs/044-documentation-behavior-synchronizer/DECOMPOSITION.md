# Decomposition: Documentation-behavior synchronizer

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Keep documentation, examples, tests, and observed behavior consistent through coordinated, evidence-backed revisions.

Invariant: Passing a test does not prove the requirement was understood, type compatibility does not guarantee behavioral compatibility, and a plausible incident story is not a verified cause.

### extractPromises

Algorithm: Extract checkable statements about inputs, outputs, defaults, errors, and supported workflows from documentation and examples. Preserve version and audience scope, distinguishing normative promises from illustrative suggestions or historical notes.

Gate: Every promise has an applicable version and source span.

Contract: ``GoalContract + PolicySelection + Relevant<DocumentationReview>[] + EvidencePage[] → StageResult<DocumentedPromise>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### probeBehavior

Algorithm: Form bounded sandbox probes using approved capabilities and representative inputs. Record environment and release identity, compare observed outcomes with the documented promise, and avoid treating one failed environment as proof of a product defect.

Gate: Probe is reproducible and isolates the behavior under review.

Contract: ``GoalContract + PolicySelection + Relevant<DocumentedPromise>[] + EvidencePage[] → StageResult<BehaviorObservation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### classifyMismatch

Algorithm: Reconcile requirements, release decisions, tests, and observed behavior to determine which artifact is wrong. Do not automatically rewrite documentation to match a bug; preserve unresolved intent questions for the responsible product or engineering owner.

Gate: Repair direction has authoritative support rather than convenience.

Contract: ``GoalContract + PolicySelection + Relevant<BehaviorObservation>[] + EvidencePage[] → StageResult<DocumentationMismatch>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### constructRepairs

Algorithm: Draft coordinated changes to pages, examples, tests, and release notes according to the reviewed repair direction. Preserve unaffected claims, update cross-links and terminology, and keep code examples executable under their stated environment.

Gate: Exact patch bases and semantic preservation checks pass.

Contract: ``GoalContract + PolicySelection + Relevant<DocumentationMismatch>[] + EvidencePage[] → StageResult<DocumentationPatch>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### verifyConsistency

Algorithm: Re-extract changed promises, rerun examples and targeted behavior probes, and inspect linked pages for stale repetitions. Publish the reviewed bundle only when its examples and claims correspond to the same release version.

Gate: No unresolved blocking mismatch or broken executable example in the released bundle.

Contract: ``GoalContract + PolicySelection + Relevant<DocumentationPatch>[] + EvidencePage[] → StageResult<DocumentationRelease>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

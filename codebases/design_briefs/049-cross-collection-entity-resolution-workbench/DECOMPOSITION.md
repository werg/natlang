# Decomposition: Cross-collection entity resolution workbench

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Resolve identities and relationships across ambiguous records without unsafe transitive merges or loss of conflicting evidence.

Invariant: A matching label is not semantic equivalence, a missing value may carry meaning, and exact aggregation is correct only after the intended population and predicates are established.

### constructCandidates

Algorithm: Generate candidate pairs using exact normalized keys, aliases, and bounded semantic retrieval. Preserve blocking provenance and include audit samples outside candidate blocks so recall failures are measurable rather than hidden by the candidate generator.

Gate: Candidate generation cannot cross access boundaries or omit known hard incompatibilities.

Contract: ``GoalContract + PolicySelection + Relevant<ResolutionProject>[] + EvidencePage[] → StageResult<IdentityCandidate>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### judgePairs

Algorithm: Compare names, attributes, chronology, and context while distinguishing identity from affiliation or succession. Treat missing attributes as unknown rather than matching, and record both positive support and hard cannot-link evidence.

Gate: Same-entity judgments require compatible identity evidence, not shared names alone.

Contract: ``GoalContract + PolicySelection + Relevant<IdentityCandidate>[] + EvidencePage[] → StageResult<IdentityJudgment>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### reconcileClusters

Algorithm: Build clusters using constrained graph operations rather than unconditional transitive closure. Before merging components, test all relevant cannot-link and temporal constraints; route inconsistent triangles to review and preserve uncertain alternatives.

Gate: No cluster contains an established cannot-link pair or impossible identity chronology.

Contract: ``GoalContract + PolicySelection + Relevant<IdentityJudgment>[] + EvidencePage[] → StageResult<IdentityCluster>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### constructCanonical

Algorithm: Assemble a canonical view without overwriting source records or selecting one source globally for every field. Resolve attributes using field-specific authority and effective time, preserve aliases and conflicting claims, and separate historical from current values.

Gate: Canonical values remain attributable and do not erase meaningful disagreement.

Contract: ``GoalContract + PolicySelection + Relevant<IdentityCluster>[] + EvidencePage[] → StageResult<CanonicalEntity>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### maintainIdentity

Algorithm: Process new evidence and reviewer corrections as reversible identity changes. Trace downstream uses, prepare split or merge migrations, and preserve old identifiers as redirects with explicit validity rather than silently changing historical joins.

Gate: Consumer impact and reversibility verified before promoting an identity revision.

Contract: ``GoalContract + PolicySelection + Relevant<CanonicalEntity>[] + EvidencePage[] → StageResult<IdentityRevision>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

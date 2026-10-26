# Decomposition: Assessment production and moderation system

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Produce valid assessments, robust rubrics, and fair moderation workflows that recognize correct unexpected answers.

Invariant: A wrong answer may reflect ambiguity rather than misconception, fluent explanation is not demonstrated competence, and accommodations must not be inferred from sensitive attributes.

### blueprintCoverage

Algorithm: Translate the assessment purpose into observable competencies and a balanced task blueprint. Distinguish intended skills from incidental reading or tool demands, and identify constraints on duration, accessibility, and allowable assistance.

Gate: Every weighted competency has a measurable evidence path.

Contract: ``GoalContract + PolicySelection + Relevant<AssessmentProject>[] + EvidencePage[] → StageResult<AssessmentBlueprint>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### generateItems

Algorithm: Create varied tasks that elicit the target evidence rather than superficial recall alone. Specify required context and resources, avoid hidden assumptions, and generate plausible alternative interpretations for subsequent review.

Gate: Item is answerable from supplied knowledge and permitted resources.

Contract: ``GoalContract + PolicySelection + Relevant<AssessmentBlueprint>[] + EvidencePage[] → StageResult<AssessmentItem>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### constructRubrics

Algorithm: Define observable scoring criteria and partial-credit boundaries, separating conceptual correctness from presentation preferences. Include correct alternative methods and common errors, and state how ambiguity or missing evidence should be handled.

Gate: Rubric does not penalize a valid solution merely for differing from the model answer.

Contract: ``GoalContract + PolicySelection + Relevant<AssessmentItem>[] + EvidencePage[] → StageResult<ScoringRubric>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### challengeAssessment

Algorithm: Independently solve each item, attempt reasonable alternate readings, and test rubric consistency on borderline responses. Repair items and rubrics together, then rerun coverage and duration checks before releasing a frozen assessment edition.

Gate: No unresolved correctness or material ambiguity defect in released items.

Contract: ``GoalContract + PolicySelection + Relevant<ScoringRubric>[] + EvidencePage[] → StageResult<ItemChallenge>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### moderateAndAppeal

Algorithm: Compare scorer disagreements and appeals against the released rubric and actual response. Identify systematic rubric defects, propose scoped corrections, and apply approved changes consistently to all affected learners rather than only those who appealed.

Gate: Exact score bounds and consistent retrospective treatment under authorized moderation.

Contract: ``GoalContract + PolicySelection + Relevant<ItemChallenge>[] + EvidencePage[] → StageResult<ModerationDecision>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

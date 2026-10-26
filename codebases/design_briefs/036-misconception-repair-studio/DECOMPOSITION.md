# Decomposition: Misconception repair studio

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Diagnose likely learning obstacles and run targeted practice that demonstrates transfer rather than memorized repetition.

Invariant: A wrong answer may reflect ambiguity rather than misconception, fluent explanation is not demonstrated competence, and accommodations must not be inferred from sensitive attributes.

### interpretResponses

Algorithm: Inspect the learner’s response against the task and accepted solution space. Distinguish conceptual errors, arithmetic slips, misunderstood wording, and incomplete work; preserve alternative correct strategies rather than forcing one canonical explanation.

Gate: Observed behavior is separated from inferred learner state.

Contract: ``GoalContract + PolicySelection + Relevant<LearningCase>[] + EvidencePage[] → StageResult<ResponseObservation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### formHypotheses

Algorithm: Generate a small set of plausible explanations for the observed pattern, including task ambiguity and accidental error. Identify a question whose answers would distinguish these explanations instead of treating one wrong response as a stable misconception.

Gate: No diagnosis based on identity or a single uninformative response.

Contract: ``GoalContract + PolicySelection + Relevant<ResponseObservation>[] + EvidencePage[] → StageResult<MisconceptionHypothesis>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### generateDiagnostics

Algorithm: Construct a short diagnostic task varying the relevant conceptual feature while controlling unrelated difficulty. Include an independently checked answer space and accessible wording; avoid hints that reveal which hypothesis the system expects to confirm.

Gate: Diagnostic outcomes discriminate hypotheses and do not introduce unlearned prerequisites.

Contract: ``GoalContract + PolicySelection + Relevant<MisconceptionHypothesis>[] + EvidencePage[] → StageResult<DiagnosticTask>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### planRepair

Algorithm: Select a suitable explanation and contrastive examples, then create practice with gradually reduced scaffolding. Adapt to observed responses without simply repeating the original answer, and escalate persistent ambiguity to the instructor.

Gate: Explanations and examples are correct under an independent subject-matter check.

Contract: ``GoalContract + PolicySelection + Relevant<DiagnosticTask>[] + EvidencePage[] → StageResult<LearningIntervention>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### verifyTransfer

Algorithm: Assess performance on new tasks that change surface details and require the intended concept. Distinguish independent transfer from success under hints, update the learner model conservatively, and retain evidence for instructor review.

Gate: Mastery requires novel unassisted evidence under the supplied competency standard.

Contract: ``GoalContract + PolicySelection + Relevant<LearningIntervention>[] + EvidencePage[] → StageResult<TransferFinding>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

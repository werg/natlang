# Marketplace dispute case builder — semantic decomposition

**Brief:** `applications/03-customer-service-and-commercial-casework/013-marketplace-dispute-case-builder.md`
**Domain:** `Customer service and commercial casework`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `reconstructTimeline` → `DisputeEvent

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Align listings, messages, shipping records, and submissions without equating timestamps with event occurrence. Attribute disputed statements to their speakers, preserve edited listings, and mark contradictions rather than manufacturing a single settled narrative.
**Acceptance gate:** Every event distinguishes asserted facts from independently corroborated records.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `identifyQuestions` → `DisputedQuestion

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<DisputeEvent>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Determine which factual questions would change the rule-based outcome. Separate immaterial disagreements from decisive ones, identify the applicable evidence burden, and formulate neutral questions that either party can answer with concrete records.
**Acceptance gate:** Questions do not assume guilt, motive, or facts not established in evidence.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `assessEvidence` → `EvidenceFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<DisputedQuestion>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Evaluate relevance, consistency, and provenance of each submitted item against a specific question. Distinguish evidence authenticity checks from semantic support, and discount multiple copies of the same record without dismissing independent corroboration.
**Acceptance gate:** No private-party submission is treated as conclusive solely because it is detailed.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `applyRules` → `DisputeRecommendation

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<EvidenceFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Apply supplied marketplace rules to established facts and explicitly conditional alternatives for unresolved facts. Explain how the outcome changes under plausible resolutions and route cases outside delegated interpretation to authorized reviewers.
**Acceptance gate:** Outcome follows the selected rule edition; uncertainty cannot silently become an adverse finding.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `prepareDisputeDossier` → `ReviewDossier

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<DisputeRecommendation>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Assemble a balanced dossier with evidence, disputed questions, reasoning summary, and each party’s response opportunity. Redact unnecessary personal information while preserving contestable evidence and record the reviewer’s actual disposition separately from the recommendation.
**Acceptance gate:** Both relevant positions represented; disclosure scope and response windows satisfied.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A seller edits the listing after purchase: bind review to the transaction-time edition.
- One party uploads the other’s address publicly: restrict it without deleting evidentiary provenance.
- A deadline falls during a service outage: apply the configured extension rule and notify both parties.
- A payment reversal occurs externally: reconcile the ledger before recommending an additional remedy.
- An appeal contains new evidence: reopen only affected findings and preserve the original decision.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.

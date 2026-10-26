# Amendment impact engine — semantic decomposition

**Brief:** `applications/06-contracts-policies-and-information-governance/027-amendment-impact-engine.md`
**Domain:** `Contracts, policies, and information governance`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `identifyTargets` → `AmendmentTarget

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read amendment language against the referenced agreement edition. Resolve section references, defined terms, and indirect targets; distinguish correction of wording from a substantive change and retain ambiguous references for review rather than choosing the nearest heading.
**Acceptance gate:** Target identity and base edition established before transformation.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `interpretScope` → `AmendmentScope

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<AmendmentTarget>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Determine effective dates, covered transactions, parties, and explicit exclusions. Separate signature date from effective date, preserve grandfathered cases, and identify whether the amendment applies retroactively only when the supplied text supports that interpretation.
**Acceptance gate:** No application outside established temporal and entity scope.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `composeObligations` → `AmendedObligation

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<AmendmentScope>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Construct the amended obligation by applying the specific operation to its target. Preserve untouched exceptions, prerequisites, definitions, and cross-references; compare the result against counterexamples that would expose accidental broadening or deletion.
**Acceptance gate:** Protected conditions remain equivalent; every removed condition is authorized by the amendment.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `traceImpacts` → `AmendmentImpact

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<AmendedObligation>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Traverse dependency edges from changed obligation facets, not the whole agreement. Assess which open tasks, approvals, notices, and future schedules become stale while retaining completed actions that were valid under their historical effective rules.
**Acceptance gate:** Exact dependency reachability followed by semantic materiality; no global invalidation by default.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `releaseRevision` → `AmendmentRelease

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<AmendmentImpact>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Present a before-and-after obligation diff and affected-case replay to qualified reviewers. Freeze the approved interpretation, rebase active cases at controlled checkpoints, and preserve old versions for disputes, audit, and rollback of uncommitted changes.
**Acceptance gate:** No in-flight commitment changes silently; active cases record their interpretation migration decision.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- An amendment references an old section number: resolve by the executed edition and flag ambiguity.
- A retroactive amendment arrives after billing: propose a reviewed correction rather than rewriting history.
- Two amendments overlap: construct a precedence question instead of assuming upload order.
- A task is approved before cutover but executes after it: revalidate applicable scope at commit time.
- An unaffected exception disappears in drafting: block release with a preservation test.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.

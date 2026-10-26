# Negotiation clause package designer — semantic decomposition

**Brief:** `applications/06-contracts-policies-and-information-governance/028-negotiation-clause-package-designer.md`
**Domain:** `Contracts, policies, and information governance`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `recoverPositions` → `NegotiationPosition

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Translate stakeholder goals and the approved playbook into issue-level positions. Distinguish mandatory limits from preferences, identify dependencies between issues, and preserve disagreements that require a mandate decision before drafting a proposal.
**Acceptance gate:** Every allowed concession has a supplied authority basis.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `mapClauseInteractions` → `ClauseDependency

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<NegotiationPosition>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Inspect definitions, remedies, caps, exceptions, and cross-references together. Identify where a concession changes another clause’s effect, including apparently local edits that broaden an exception or undermine a negotiated protection.
**Acceptance gate:** Interaction claims have source references and a concrete example of changed behavior.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `generatePackages` → `ClausePackage

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ClauseDependency>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Draft materially distinct packages that exchange permitted concessions while preserving required protections. Explain package-level tradeoffs and include fallback options; avoid optimizing individual clauses independently when their effects depend on the surrounding agreement.
**Acceptance gate:** All hard playbook limits pass and no protection is claimed when another clause negates it.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `reconcileRedlines` → `RedlineFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ClausePackage>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Interpret counterparty edits in context, separating stylistic changes from altered obligations or risk allocation. Recompute package interactions, identify conditional acceptance, and route edits beyond the mandate to qualified reviewers.
**Acceptance gate:** No unreviewed material change hidden as editorial cleanup.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `prepareOffer` → `ClauseOffer

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<RedlineFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Assemble a coherent draft, issue list, and internal rationale with confidential fallback positions excluded from the external copy. Bind approval to the exact package and record which issues remain open rather than presenting a partial agreement as final.
**Acceptance gate:** External artifact contains no internal negotiation limits; current scoped approval required.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A defined term changes globally: re-evaluate every consumer, not only the edited clause.
- Internal fallback notes are in comments: strip and verify them before external export.
- A counterparty accepts one clause conditionally: preserve the condition in the package state.
- An approver signs an earlier draft: reject release of the newer hash.
- A compromise introduces an undefined term: block the draft before circulation.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.

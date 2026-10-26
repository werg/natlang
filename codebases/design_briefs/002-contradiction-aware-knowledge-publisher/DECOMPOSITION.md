# Contradiction-aware knowledge publisher — semantic decomposition

**Brief:** `applications/01-organizational-knowledge-and-institutional-memory/002-contradiction-aware-knowledge-publisher.md`
**Domain:** `Organizational knowledge and institutional memory`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `indexClaims` → `ScopedClaim

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Extract independently checkable claims with qualifications and page context. Resolve entity references conservatively, attach author and effective period, and keep instructions separate from descriptions of historical behavior or quoted third-party views.
**Acceptance gate:** Claims retain every material negation, exception, and audience qualifier.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `compareClaims` → `ClaimConflict

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ScopedClaim>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Generate candidates through shared entities and facets, then compare meanings under aligned scope and time. Classify coexistence as legitimate when claims describe different editions or populations; record the minimum incompatible propositions when a real conflict exists.
**Acceptance gate:** Pair labels checked against scoped truth fixtures; duplicate pages add no authority.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `resolveAuthority` → `AuthorityDecision

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ClaimConflict>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Consult the configured publication hierarchy and explicit supersession records. Prefer authoritative material only inside its remit; if equally authoritative sources disagree, formulate a bounded clarification rather than selecting the newest or most fluent text.
**Acceptance gate:** Unresolved decisive authority blocks publication of the disputed claim.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `composePatchSet` → `PagePatch

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<AuthorityDecision>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Draft the smallest coordinated changes that resolve the selected conflicts. Preserve unrelated claims and local audience framing; update examples, cross-links, and summaries that would otherwise repeat the corrected mistake.
**Acceptance gate:** Patch applies to exact base hash; unchanged protected claims remain semantically equivalent.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `verifyPublication` → `PublicationCheck

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<PagePatch>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Re-extract claims from the proposed page set and repeat comparison around changed topics. Inspect whether simplification removed exceptions; build a review packet with residual disagreements and publish only after required owners approve the same patch set.
**Acceptance gate:** Zero new contradiction in protected fixtures and no unresolved blocking conflict.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A page is edited during review: rebase only nonconflicting sections and require renewed approval for changed meaning.
- A translated page lags its source: flag edition mismatch rather than calling the translation false.
- A link points to an archived policy: preserve the archive but redirect current guidance explicitly.
- A denied source vanishes from search: invalidate derived claims without leaking its title.
- A release partially fails: retain a repair queue and do not mark the bundle fully published.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.

# Organizational knowledge integration workbench — semantic decomposition

**Brief:** `applications/01-organizational-knowledge-and-institutional-memory/004-organizational-knowledge-integration-workbench.md`
**Domain:** `Organizational knowledge and institutional memory`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `mapVocabularies` → `TermSense

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Extract terms from actual usage rather than glossary headings alone. Separate multiple senses within one organization and identify distinctions that affect who acts, which records exist, or what counts as completion.
**Acceptance gate:** Every sense has contextual examples; identical spelling does not establish equivalence.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `alignConcepts` → `ConceptAlignment

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<TermSense>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare operational meanings and counterexamples for candidate terms. Establish equivalence only when obligations and consequences align; represent partial overlap explicitly and retain organization-specific variants where unification would lose behavior.
**Acceptance gate:** Bidirectional substitution tests pass before a same relation is accepted.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `reconcileRoles` → `RoleInterface

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ConceptAlignment>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare actual decision rights, escalation paths, and handoff expectations. Identify duplicated ownership and gaps between roles; propose responsibility agreements without assuming that similar job titles imply interchangeable authority.
**Acceptance gate:** Every required responsibility has either an accepted owner or an explicit unresolved gap.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `planMigration` → `IntegrationWave

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<RoleInterface>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Group compatible vocabulary and procedure changes into reversible waves. Preserve source identities, specify conditional transformations, and defer records whose meaning cannot be preserved under the target model without an explicit extension.
**Acceptance gate:** Exact referential-integrity checks and semantic round-trip tests on representative records.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `validateInterfaces` → `InterfaceScenario

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<IntegrationWave>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Replay cross-team cases using proposed terminology and responsibilities. Look for steps both teams assume the other performs; turn failures into interface decisions and rerun only affected scenarios before approving a migration wave.
**Acceptance gate:** No unowned critical handoff and no source distinction silently discarded.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- The same acronym names different systems: require context-bound expansion in merged search results.
- A former organization must remain searchable: retain aliases and provenance rather than rewriting history.
- A merger team lacks personnel access: use role-level representations with restricted identity resolution.
- Cutover is postponed: shift execution dates without rewriting already effective decisions.
- A team rejects a shared definition: preserve the disagreement and its operational boundary.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.

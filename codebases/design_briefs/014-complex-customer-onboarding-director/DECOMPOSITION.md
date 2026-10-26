# Complex customer onboarding director — semantic decomposition

**Brief:** `applications/03-customer-service-and-commercial-casework/014-complex-customer-onboarding-director.md`
**Domain:** `Customer service and commercial casework`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `clarifyOutcomes` → `OnboardingOutcome

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Convert broad goals into observable outcomes and confirm what onboarding does not include. Distinguish an executive aspiration from an agreed deliverable, and identify decisions requiring customer authority before technical work begins.
**Acceptance gate:** Each committed outcome has an owner and a verifiable acceptance condition.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `mapEnvironment` → `EnvironmentConstraint

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<OnboardingOutcome>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Reconcile customer architecture notes, sample data, security requirements, and available integration capabilities. Separate verified compatibility from assumptions and formulate narrow discovery tasks for unknown interfaces rather than promising unsupported integrations.
**Acceptance gate:** No production credential in model context; unverified capability remains a blocker.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `constructPlan` → `OnboardingTask

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<EnvironmentConstraint>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Decompose outcomes into configuration, data, training, and acceptance tasks. Order dependencies, allocate ownership explicitly, and create reversible pilots before broad changes; identify shared prerequisites that would otherwise generate duplicated requests.
**Acceptance gate:** Task graph is acyclic and every outcome is covered by an executable path.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `interpretProgress` → `ProgressFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<OnboardingTask>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Read system events and participant updates against each task’s completion contract. Distinguish an attempted import from a reconciled migration and attendance from demonstrated setup capability; request the smallest missing verification artifact.
**Acceptance gate:** Completion requires evidence, not a percentage reported in free text.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `replanAndAccept` → `OnboardingRevision

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ProgressFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** When dependencies fail or objectives change, construct feasible alternatives and disclose scope, timing, and effort consequences. Preserve accepted work, obtain approval for material changes, and replay end-to-end outcomes before declaring onboarding complete.
**Acceptance gate:** Revised plan preserves mandatory requirements or records explicit accepted scope reductions.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A customer administrator leaves: transfer task ownership without reusing their credentials.
- Sample data works but production data differs: stage a representative validation before full import.
- A vendor integration is unavailable: offer a documented temporary path without pretending the original scope is complete.
- Training attendees differ from system owners: record who can actually perform required tasks.
- A launch date changes: preserve contractual dates separately from internal targets.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.

# Product opportunity portfolio manager — semantic decomposition

**Brief:** `applications/02-customer-research-and-product-intelligence/008-product-opportunity-portfolio-manager.md`
**Domain:** `Customer research and product intelligence`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `reconstructNeeds` → `CustomerNeed

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Reconcile customer requests into underlying outcomes and constraints. Separate a suggested implementation from the need it serves; preserve segments whose superficially similar requests arise from different workflows and require different interventions.
**Acceptance gate:** Every need has attributable evidence; invented segments and unsupported market sizes are forbidden.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `generateInterventions` → `Intervention

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<CustomerNeed>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Propose materially different ways to improve each outcome, including process, service, and product changes. Explain the mechanism, affected users, and expected tradeoffs; avoid generating cosmetic variants that all depend on the same untested assumption.
**Acceptance gate:** Candidate set includes a credible low-change baseline and distinct mechanisms.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `estimateDependencies` → `OpportunityEstimate

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<Intervention>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Translate each intervention into bounded work packages, request estimates from authorized sources, and identify shared enabling work. Preserve cost ranges and unknowns rather than inventing precision; inspect whether dependencies make apparently independent projects coupled.
**Acceptance gate:** Shared work is costed once; missing estimates remain explicit and block firm commitments.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `constructPortfolios` → `PortfolioOption

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<OpportunityEstimate>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Submit eligible options and exact constraints to a portfolio solver, then interpret feasible alternatives against the supplied strategy. Explain who benefits and what is deferred; generate contingency portfolios for major assumption changes rather than a single opaque ranking.
**Acceptance gate:** Exact budget and capacity feasibility; ranking rationale uses supplied priorities only.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `planLearning` → `OpportunityExperiment

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<PortfolioOption>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Choose the cheapest credible experiment capable of changing a portfolio decision. State how positive, negative, and inconclusive results would alter the selected work, and schedule learning before irreversible dependent commitments wherever feasible.
**Acceptance gate:** An experiment has an actionable decision rule and does not merely collect agreeable feedback.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A sponsor changes priorities mid-cycle: fork the scenario and preserve the approved baseline.
- Two projects claim the same benefit: identify overlap instead of summing both estimates.
- A dependency estimate is missing: show a contingent portfolio, not false feasibility.
- A customer quote supports several needs: preserve one source and avoid multiplying evidence strength.
- An experiment is inconclusive: retain the uncertainty rather than treating it as a failed idea.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.

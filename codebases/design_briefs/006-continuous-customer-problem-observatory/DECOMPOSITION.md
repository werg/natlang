# Continuous customer-problem observatory — semantic decomposition

**Brief:** `applications/02-customer-research-and-product-intelligence/006-continuous-customer-problem-observatory.md`
**Domain:** `Customer research and product intelligence`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `extractEpisodes` → `ProblemEpisode

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read enough surrounding conversation to identify the attempted job, the obstacle, and its consequence. Separate a product complaint from a requested feature, and split independently resolvable problems while retaining links to the original interaction.
**Acceptance gate:** Every episode preserves the customer's actual situation and does not invent severity.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `linkIncidents` → `IncidentGroup

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ProblemEpisode>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare accounts for the same customer, event, and time window. Group follow-ups and copied messages into one incident while retaining repeat failures that actually occurred on different occasions; leave identity ambiguity unresolved.
**Acceptance gate:** Counts distinguish messages, incidents, and customers; no transitive merge across incompatible events.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `maintainProblemMap` → `ProblemConcept

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<IncidentGroup>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Cluster by the failed outcome and mechanism rather than shared vocabulary. Propose new concepts when existing boundaries do not fit, preserve meaningful subtypes, and review counterexamples before moving historical episodes between concepts.
**Acceptance gate:** Reclassification preserves raw episode IDs and records concept-version lineage.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `computeTrends` → `TrendSlice

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ProblemConcept>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Define an observable population and comparable time windows before invoking exact aggregation. Interpret changes in collection channels and exposure; distinguish a rise in complaints from a rise in the complaint rate among eligible customers.
**Acceptance gate:** Numerators are subsets of denominators; missing exposure forbids unsupported rate claims.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `commissionInvestigation` → `InvestigationBrief

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<TrendSlice>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Select unresolved explanations that would change a product decision. Formulate a bounded investigation with relevant cohorts, contradictory cases, and evidence needed to distinguish mechanisms rather than recommending the loudest requested feature.
**Acceptance gate:** Each task answers a named uncertainty and cites both support and known counterevidence.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A single outage creates thousands of tickets: separate event volume from independent problem discovery.
- A survey channel disappears: mark trend comparability as broken rather than reporting improvement.
- Consent is withdrawn: remove attributable quotations and recompute restricted derivatives.
- Customer IDs change after account migration: version identity mappings before updating counts.
- A topic is renamed: preserve stable concept IDs so dashboards do not reset history.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.

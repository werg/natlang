# Disclosure and redaction review workspace — semantic decomposition

**Brief:** `applications/06-contracts-policies-and-information-governance/030-disclosure-and-redaction-review-workspace.md`
**Domain:** `Contracts, policies, and information governance`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `interpretScope` → `DisclosureScope

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read the request, authorization, and supplied disclosure policy together. Determine recipient, purpose, relevant time span, and permitted material; flag conflicting instructions rather than treating the broadest request as sufficient authority.
**Acceptance gate:** Scope and recipient authority are established before document access or export.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `selectMaterial` → `DisclosureSelection

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<DisclosureScope>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Identify responsive content at passage level, retaining enough context for accurate interpretation. Separate nonresponsive material from restricted responsive material so reviewers can see whether an omission is about relevance or permission.
**Acceptance gate:** No whole-document inclusion merely because one passage is relevant.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `proposeRedactions` → `RedactionProposal

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<DisclosureSelection>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Apply supplied rules to personal, confidential, privileged, or third-party material. Propose the smallest redaction that protects the restricted information while preserving the permitted claim; identify cases requiring qualified interpretation rather than making final legal judgments.
**Acceptance gate:** Each redaction has a rule and precise span; over-redaction is evaluated separately from leakage.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `inspectComposition` → `DisclosureRisk

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<RedactionProposal>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Review the proposed bundle for information revealed by combining otherwise permissible fragments, filenames, metadata, and repeated identifiers. Distinguish plausible reconstruction from speculative risk and propose targeted changes for reviewer approval.
**Acceptance gate:** Bundle-level checks include metadata and cross-document references, not text alone.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `renderAndRelease` → `DisclosureManifest

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<DisclosureRisk>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Render irreversible redactions through a trusted host service, inspect extracted text and metadata, and assemble a manifest. Bind approval and delivery to the exact recipient and artifact hashes, retaining a controlled review copy and disclosure receipt.
**Acceptance gate:** No recoverable redacted text, hidden layers, or wrong-recipient delivery.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A PDF contains hidden OCR text: sanitize all text layers and verify extraction after rendering.
- A filename contains a restricted identity: rename the export and remove embedded metadata.
- A recipient changes after approval: invalidate delivery authorization.
- A redaction removes a necessary qualifier: revise the surrounding permitted text without inventing meaning.
- A disclosure is challenged: retain an auditable mapping from released artifacts to reviewed source spans.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.

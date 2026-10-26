---
description: Prepare or reconcile an identity-bound referral packet against recipient requirements and exact delivery state.
args: { dossier: ReferralDossier; change: PacketChange; coordinator: ReferralCoordinator }
returns: ReferralProposal
---
Keep the packet within its patient, episode and receiving-service scope. A shared name is never an identity match. Carry clinician priority from its authoritative reference unchanged; do not re-triage it.

Bind identity = `bindReferralIdentity(dossier, change)`. If identity is uncertain or mismatched, block document matching and release. Do not use another chart's records.

Bind checklist = `applyRecipientChecklistDelta(dossier, change, identity)` against the current authorized checklist. Revalidate only pending requirements whose rule version changed; keep unaffected findings intact.

Bind documents = `inspectDocumentCompleteness(dossier, change, identity, checklist)`. For each item, distinguish valid, unsigned, expired, wrong-episode, identity-uncertain and missing. A file's presence alone does not satisfy its requirement.

Bind requests = `requestMissingMaterial(dossier, documents)` from the role that owns the missing evidence or signature. Do not request unrelated history.

Bind transfer = `reconcileReferralTransfer(dossier, change, identity, documents, requests)`. Dispatch only with current authorization, verified identity, closed checklist gaps and episode-limited disclosure. If acknowledgment is lost, read recipient status before resending sensitive records. Unknown delivery is not safe to resend.

Bind effects = transfer.effects or an empty array. Construct the complete `ReferralProposal`: set application to its declared literal; copy the exact `identity`, updated `checklist`, all document findings, current requirement-level `dossier.gaps`, requests returned by `requestMissingMaterial`, and the exact transfer ledger. Copy `dossier.checkpoint.priorityRef`, referral, episode scope and disclosure refs unchanged into a complete resumeState; replace only its identity, checklist, documents, requests and transfer with the outputs just reconciled. Set effects to the exact typed transfer effects. Set blockers to the union of identity uncertainty, unresolved requirements, unsafe document findings, stale checklist authority and transfer blockers. Mark blocked when identity is not verified, a hard requirement is unresolved, or transfer has a blocker; otherwise mark ready-for-review. Set actionBoundary to state that only a separately authorized referral adapter may transmit the exact episode-limited packet and only recipient readback can confirm receipt. A prepared or dispatched packet is not recipient-acknowledged until an exact receipt exists. Return every ReferralProposal field explicitly; no branch returns only status or a prose handoff.

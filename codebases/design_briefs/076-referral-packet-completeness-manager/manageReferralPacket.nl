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

Return `ReferralProposal` for `coordinator.id` with clinician priority unchanged, requirement-level gaps and a resumable transfer ledger. A prepared or dispatched packet is not recipient-acknowledged until an exact receipt exists.

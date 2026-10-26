---
args: { candidate: AmendmentImpact, decisions: DecisionsAmendmentRelease, evidence: EvidencePage[] }
returns: DraftAmendmentRelease
---
1. Construct an evidence-backed AmendmentRelease record from established atoms, using the fields declared for AmendmentRelease in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Present a before-and-after obligation diff and affected-case replay to qualified reviewers. Freeze the approved interpretation, rebase active cases at controlled checkpoints, and preserve old versions for disputes, audit, and rollback of uncommitted changes.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

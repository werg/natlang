---
args: { candidate: Commitment, decisions: DecisionsWorkStatus, evidence: EvidencePage[] }
returns: DraftWorkStatus
---
1. Construct an evidence-backed WorkStatus record from established atoms, using the fields declared for WorkStatus in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Match outputs and acknowledgments to each promised outcome. A sent file is not accepted delivery when acceptance was required; cancellation must come from an authorized party, and incomplete evidence produces a status question.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

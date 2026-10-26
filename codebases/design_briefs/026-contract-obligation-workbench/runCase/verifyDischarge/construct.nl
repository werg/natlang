---
args: { candidate: ObligationTask, decisions: DecisionsObligationStatus, evidence: EvidencePage[] }
returns: DraftObligationStatus
---
1. Construct an evidence-backed ObligationStatus record from established atoms, using the fields declared for ObligationStatus in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare completed actions with the actual duty and any acceptance condition. Distinguish performance from waiver or settlement, record reviewer-approved interpretations, and keep disputed discharge visible rather than closing on an internal completion flag.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

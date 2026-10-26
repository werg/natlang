---
args: { candidate: RecoveryAction, decisions: DecisionsRecoveryClosure, evidence: EvidencePage[] }
returns: DraftRecoveryClosure
---
1. Construct an evidence-backed RecoveryClosure record from established atoms, using the fields declared for RecoveryClosure in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare actual outcomes with every agreed remedy and earlier promise. A refund request or shipment label is not completion; retain unresolved components and explain partial closure without resetting the entire case history.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

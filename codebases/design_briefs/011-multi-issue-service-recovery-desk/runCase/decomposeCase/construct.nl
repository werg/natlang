---
args: { candidate: RecoveryCase, decisions: DecisionsServiceIssue, evidence: EvidencePage[] }
returns: DraftServiceIssue
---
1. Construct an evidence-backed ServiceIssue record from established atoms, using the fields declared for ServiceIssue in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Reconstruct the timeline across messages and split failures only when they can be resolved independently. Retain earlier promises and interactions between issues, such as a late replacement making a promotional remedy no longer useful.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

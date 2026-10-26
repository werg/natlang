---
args: { candidate: ClausePackage, decisions: DecisionsRedlineFinding, evidence: EvidencePage[] }
returns: DraftRedlineFinding
---
1. Construct an evidence-backed RedlineFinding record from established atoms, using the fields declared for RedlineFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Interpret counterparty edits in context, separating stylistic changes from altered obligations or risk allocation. Recompute package interactions, identify conditional acceptance, and route edits beyond the mandate to qualified reviewers.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

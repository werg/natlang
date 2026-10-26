---
args: { candidate: ReconciliationBreak, decisions: DecisionsCloseTask, evidence: EvidencePage[] }
returns: DraftCloseTask
---
1. Construct an evidence-backed CloseTask record from established atoms, using the fields declared for CloseTask in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Translate each consequential unresolved break into a bounded investigation with an accountable owner. Specify evidence capable of proving the explanation, prioritize dependencies on other close tasks, and expose unresolved issues before sign-off deadlines.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

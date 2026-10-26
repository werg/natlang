---
args: { candidate: ContainmentPlan, decisions: DecisionsCorrectiveAction, evidence: EvidencePage[] }
returns: DraftCorrectiveAction
---
1. Construct an evidence-backed CorrectiveAction record from established atoms, using the fields declared for CorrectiveAction in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Evaluate the supplier’s explanation against incident evidence and rival mechanisms. Separate immediate correction from prevention of recurrence, identify testable claims, and request evidence that the proposed change addresses the actual mechanism rather than just the symptoms.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

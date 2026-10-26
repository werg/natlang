---
args: { candidate: CorrectiveAction, decisions: DecisionsIncidentClosure, evidence: EvidencePage[] }
returns: DraftIncidentClosure
---
1. Construct an evidence-backed IncidentClosure record from established atoms, using the fields declared for IncidentClosure in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Check implemented actions, release criteria, and subsequent observations. Preserve residual risk and monitoring obligations; reopen the incident when a materially similar recurrence contradicts the closure rationale rather than treating it as a fresh unrelated ticket.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

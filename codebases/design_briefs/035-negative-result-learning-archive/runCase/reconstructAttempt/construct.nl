---
args: { candidate: NegativeResultCase, decisions: DecisionsResearchAttempt, evidence: EvidencePage[] }
returns: DraftResearchAttempt
---
1. Construct an evidence-backed ResearchAttempt record from established atoms, using the fields declared for ResearchAttempt in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read plans, execution logs, and researcher notes to reconstruct the attempted test. Separate intended from actual procedure and identify incomplete steps, measurement failures, or deviations that limit what the result can establish.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

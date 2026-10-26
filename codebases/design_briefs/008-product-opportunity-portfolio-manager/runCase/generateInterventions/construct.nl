---
args: { candidate: CustomerNeed, decisions: DecisionsIntervention, evidence: EvidencePage[] }
returns: DraftIntervention
---
1. Construct an evidence-backed Intervention record from established atoms, using the fields declared for Intervention in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Propose materially different ways to improve each outcome, including process, service, and product changes. Explain the mechanism, affected users, and expected tradeoffs; avoid generating cosmetic variants that all depend on the same untested assumption.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

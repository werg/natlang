---
args: { candidate: AssumptionBinding, decisions: DecisionsChangeImpact, evidence: EvidencePage[] }
returns: DraftChangeImpact
---
1. Construct an evidence-backed ChangeImpact record from established atoms, using the fields declared for ChangeImpact in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare old and new assertions under their dates, scope, and authority. Determine which premise is actually invalidated and retain the part of the decision whose conditions still hold; never treat a synonym edit as a substantive reversal.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

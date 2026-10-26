---
args: { candidate: AmendedObligation, decisions: DecisionsAmendmentImpact, evidence: EvidencePage[] }
returns: DraftAmendmentImpact
---
1. Construct an evidence-backed AmendmentImpact record from established atoms, using the fields declared for AmendmentImpact in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Traverse dependency edges from changed obligation facets, not the whole agreement. Assess which open tasks, approvals, notices, and future schedules become stale while retaining completed actions that were valid under their historical effective rules.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

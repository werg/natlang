---
args: { candidate: ChangeIntent, decisions: DecisionsChangeConsequence, evidence: EvidencePage[] }
returns: DraftChangeConsequence
---
1. Construct an evidence-backed ChangeConsequence record from established atoms, using the fields declared for ChangeConsequence in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Consult authoritative entitlement and billing services for permitted dates and amounts, then explain consequences in plain language. Separate immediate access effects from future billing and data deletion; keep uncertain external balances explicitly unresolved.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

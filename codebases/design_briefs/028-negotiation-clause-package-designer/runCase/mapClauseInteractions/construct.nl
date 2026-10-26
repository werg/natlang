---
args: { candidate: NegotiationPosition, decisions: DecisionsClauseDependency, evidence: EvidencePage[] }
returns: DraftClauseDependency
---
1. Construct an evidence-backed ClauseDependency record from established atoms, using the fields declared for ClauseDependency in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Inspect definitions, remedies, caps, exceptions, and cross-references together. Identify where a concession changes another clause’s effect, including apparently local edits that broaden an exception or undermine a negotiated protection.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

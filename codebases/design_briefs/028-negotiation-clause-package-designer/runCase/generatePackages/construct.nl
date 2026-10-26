---
args: { candidate: ClauseDependency, decisions: DecisionsClausePackage, evidence: EvidencePage[] }
returns: DraftClausePackage
---
1. Construct an evidence-backed ClausePackage record from established atoms, using the fields declared for ClausePackage in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Draft materially distinct packages that exchange permitted concessions while preserving required protections. Explain package-level tradeoffs and include fallback options; avoid optimizing individual clauses independently when their effects depend on the surrounding agreement.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

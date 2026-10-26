---
args: { candidate: ContractObligation, decisions: DecisionsObligationRelation, evidence: EvidencePage[] }
returns: DraftObligationRelation
---
1. Construct an evidence-backed ObligationRelation record from established atoms, using the fields declared for ObligationRelation in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Interpret cross-references and interactions among provisions under the supplied authority hierarchy. Distinguish a narrow exception from replacement of an entire obligation and preserve unresolved conflicts for qualified review instead of guessing a priority rule.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

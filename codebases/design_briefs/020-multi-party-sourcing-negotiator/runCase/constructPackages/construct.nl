---
args: { candidate: SupplierOffer, decisions: DecisionsSourcingPackage, evidence: EvidencePage[] }
returns: DraftSourcingPackage
---
1. Construct an evidence-backed SourcingPackage record from established atoms, using the fields declared for SourcingPackage in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Use exact allocation services to enumerate feasible combinations, then compare operational resilience and semantic fit under supplied priorities. Consider split supply and alternative delivery patterns without creating allocations that suppliers have not offered capacity to fulfill.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

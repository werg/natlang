---
args: { candidate: SourcingRequirement, decisions: DecisionsSupplierOffer, evidence: EvidencePage[] }
returns: DraftSupplierOffer
---
1. Construct an evidence-backed SupplierOffer record from established atoms, using the fields declared for SupplierOffer in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Reconcile quotations into comparable units and scopes while preserving bundles, exclusions, and conditional discounts. Distinguish an indicative quote from a binding offer and expose missing service or logistics costs rather than assuming comparability.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

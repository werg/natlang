---
args: { candidate: OpportunityPortfolio, decisions: DecisionsCustomerNeed, evidence: EvidencePage[] }
returns: DraftCustomerNeed
---
1. Construct an evidence-backed CustomerNeed record from established atoms, using the fields declared for CustomerNeed in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Reconcile customer requests into underlying outcomes and constraints. Separate a suggested implementation from the need it serves; preserve segments whose superficially similar requests arise from different workflows and require different interventions.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

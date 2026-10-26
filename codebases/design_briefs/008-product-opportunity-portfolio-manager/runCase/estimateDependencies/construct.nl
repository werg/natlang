---
args: { candidate: Intervention, decisions: DecisionsOpportunityEstimate, evidence: EvidencePage[] }
returns: DraftOpportunityEstimate
---
1. Construct an evidence-backed OpportunityEstimate record from established atoms, using the fields declared for OpportunityEstimate in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Translate each intervention into bounded work packages, request estimates from authorized sources, and identify shared enabling work. Preserve cost ranges and unknowns rather than inventing precision; inspect whether dependencies make apparently independent projects coupled.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

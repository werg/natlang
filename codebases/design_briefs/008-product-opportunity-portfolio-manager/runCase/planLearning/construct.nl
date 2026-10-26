---
args: { candidate: PortfolioOption, decisions: DecisionsOpportunityExperiment, evidence: EvidencePage[] }
returns: DraftOpportunityExperiment
---
1. Construct an evidence-backed OpportunityExperiment record from established atoms, using the fields declared for OpportunityExperiment in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Choose the cheapest credible experiment capable of changing a portfolio decision. State how positive, negative, and inconclusive results would alter the selected work, and schedule learning before irreversible dependent commitments wherever feasible.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

---
args: { candidate: OpportunityEstimate, decisions: DecisionsPortfolioOption, evidence: EvidencePage[] }
returns: DraftPortfolioOption
---
1. Construct an evidence-backed PortfolioOption record from established atoms, using the fields declared for PortfolioOption in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Submit eligible options and exact constraints to a portfolio solver, then interpret feasible alternatives against the supplied strategy. Explain who benefits and what is deferred; generate contingency portfolios for major assumption changes rather than a single opaque ranking.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

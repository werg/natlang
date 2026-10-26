---
args: { candidate: TrendSlice, decisions: DecisionsInvestigationBrief, evidence: EvidencePage[] }
returns: DraftInvestigationBrief
---
1. Construct an evidence-backed InvestigationBrief record from established atoms, using the fields declared for InvestigationBrief in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Select unresolved explanations that would change a product decision. Formulate a bounded investigation with relevant cohorts, contradictory cases, and evidence needed to distinguish mechanisms rather than recommending the loudest requested feature.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

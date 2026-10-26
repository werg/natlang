---
args: { candidate: ChangeImpact, decisions: DecisionsReviewPlan, evidence: EvidencePage[] }
returns: DraftReviewPlan
---
1. Construct an evidence-backed ReviewPlan record from established atoms, using the fields declared for ReviewPlan in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Traverse only necessary dependencies first, then inspect supporting edges for materiality. Group reviews sharing an unresolved premise, formulate a decision-ready question, and avoid automatically cancelling downstream work that remains independently justified.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

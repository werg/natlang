---
args: { candidate: MechanismFinding, decisions: DecisionsAnalysisOrder, evidence: EvidencePage[] }
returns: DraftAnalysisOrder
---
1. Construct an evidence-backed AnalysisOrder record from established atoms, using the fields declared for AnalysisOrder in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Convert unresolved issues into typed analysis plans with cohorts, time windows, allowed operations, and interpretation rules. Separate confirmatory follow-up from exploratory searching and require a fresh data or holdout plan where the proposed claim needs one.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

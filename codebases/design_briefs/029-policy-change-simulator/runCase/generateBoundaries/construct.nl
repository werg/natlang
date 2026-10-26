---
args: { candidate: OutcomeDifference, decisions: DecisionsBoundaryScenario, evidence: EvidencePage[] }
returns: DraftBoundaryScenario
---
1. Construct an evidence-backed BoundaryScenario record from established atoms, using the fields declared for BoundaryScenario in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Construct near-neighbor cases around changed thresholds, exceptions, and overlapping categories. Vary one consequential fact at a time, include adverse interactions, and identify scenarios whose outcome exposes an ambiguity requiring a policy decision.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.

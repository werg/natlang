---
args: { draft: DraftFabricationConcept, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated FabricationConcept field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Concept is feasible under documented shop capability and approved design constraints. Explain the decisive evidence and smallest repair target; never weaken the gate.

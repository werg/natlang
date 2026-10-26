---
args: { draft: DraftQualityActionPlan, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated QualityActionPlan field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Actions respect authority, safety prerequisites, and explicit release conditions. Explain the decisive evidence and smallest repair target; never weaken the gate.

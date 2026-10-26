---
args: { draft: DraftGardenPlan, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated GardenPlan field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Known hard constraints and approved garden rules remain satisfied. Explain the decisive evidence and smallest repair target; never weaken the gate.

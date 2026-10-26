---
args: { draft: DraftHarvestPlan, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated HarvestPlan field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact resource feasibility and quantity conservation. Explain the decisive evidence and smallest repair target; never weaken the gate.

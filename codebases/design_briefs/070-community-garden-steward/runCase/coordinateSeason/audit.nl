---
args: { draft: DraftGardenTask, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated GardenTask field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Resource scheduling and member consent remain current through replanning. Explain the decisive evidence and smallest repair target; never weaken the gate.

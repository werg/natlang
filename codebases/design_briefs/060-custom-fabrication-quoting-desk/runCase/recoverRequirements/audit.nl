---
args: { draft: DraftFabricationRequirement, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated FabricationRequirement field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Critical requirements are confirmed or explicitly block a firm quote. Explain the decisive evidence and smallest repair target; never weaken the gate.

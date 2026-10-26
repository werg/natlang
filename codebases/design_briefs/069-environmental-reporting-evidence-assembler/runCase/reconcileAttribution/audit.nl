---
args: { draft: DraftEnvironmentalAllocation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated EnvironmentalAllocation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact totals and no double-counting across reporting boundaries. Explain the decisive evidence and smallest repair target; never weaken the gate.

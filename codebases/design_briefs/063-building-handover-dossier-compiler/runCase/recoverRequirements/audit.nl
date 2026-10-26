---
args: { draft: DraftHandoverRequirement, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated HandoverRequirement field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every required handover item has an explicit source and acceptance condition. Explain the decisive evidence and smallest repair target; never weaken the gate.

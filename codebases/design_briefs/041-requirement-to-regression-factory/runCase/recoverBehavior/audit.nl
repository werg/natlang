---
args: { draft: DraftBehavioralRequirement, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated BehavioralRequirement field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every assertion traces to a requirement or explicitly approved interpretation. Explain the decisive evidence and smallest repair target; never weaken the gate.

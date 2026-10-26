---
args: { draft: DraftEnvironmentalClaim, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated EnvironmentalClaim field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Each quantitative and impact claim is supported at the stated scope. Explain the decisive evidence and smallest repair target; never weaken the gate.

---
args: { draft: DraftConstructionImpact, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ConstructionImpact field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Claimed downstream effects have dependency paths and explicit assumptions. Explain the decisive evidence and smallest repair target; never weaken the gate.

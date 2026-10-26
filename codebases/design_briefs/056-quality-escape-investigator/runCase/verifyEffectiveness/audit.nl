---
args: { draft: DraftQualityClosure, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated QualityClosure field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Closure requires verified effectiveness, not only completed action checkboxes. Explain the decisive evidence and smallest repair target; never weaken the gate.

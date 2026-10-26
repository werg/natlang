---
args: { draft: DraftScopeDifference, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ScopeDifference field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Each claimed change maps to a specific baseline requirement or omission. Explain the decisive evidence and smallest repair target; never weaken the gate.

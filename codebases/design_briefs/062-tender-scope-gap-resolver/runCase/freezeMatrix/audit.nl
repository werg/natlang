---
args: { draft: DraftResponsibilityMatrix, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ResponsibilityMatrix field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No silent assignment of unaccepted obligations and no uncovered mandatory function. Explain the decisive evidence and smallest repair target; never weaken the gate.

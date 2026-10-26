---
args: { draft: DraftProjectLearningGoal, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ProjectLearningGoal field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every goal has a feasible demonstration within the project scope. Explain the decisive evidence and smallest repair target; never weaken the gate.

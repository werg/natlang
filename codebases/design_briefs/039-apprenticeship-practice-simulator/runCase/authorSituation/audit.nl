---
args: { draft: DraftPracticeScenario, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated PracticeScenario field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Scenario has a feasible compliant path and no hidden prerequisite the learner cannot discover. Explain the decisive evidence and smallest repair target; never weaken the gate.

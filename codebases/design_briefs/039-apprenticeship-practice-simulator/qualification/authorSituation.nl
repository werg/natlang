---
args: { draft: DraftPracticeScenario, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to PracticeScenario: Scenario has a feasible compliant path and no hidden prerequisite the learner cannot discover. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.

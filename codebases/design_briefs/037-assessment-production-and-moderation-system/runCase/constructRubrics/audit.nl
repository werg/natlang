---
args: { draft: DraftScoringRubric, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ScoringRubric field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Rubric does not penalize a valid solution merely for differing from the model answer. Explain the decisive evidence and smallest repair target; never weaken the gate.

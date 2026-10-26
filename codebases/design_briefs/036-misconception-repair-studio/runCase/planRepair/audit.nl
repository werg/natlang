---
args: { draft: DraftLearningIntervention, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated LearningIntervention field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Explanations and examples are correct under an independent subject-matter check. Explain the decisive evidence and smallest repair target; never weaken the gate.

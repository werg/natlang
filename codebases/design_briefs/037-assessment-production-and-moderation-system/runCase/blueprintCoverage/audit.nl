---
args: { draft: DraftAssessmentBlueprint, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated AssessmentBlueprint field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every weighted competency has a measurable evidence path. Explain the decisive evidence and smallest repair target; never weaken the gate.

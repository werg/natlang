---
args: { draft: DraftAssessmentItem, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated AssessmentItem field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Item is answerable from supplied knowledge and permitted resources. Explain the decisive evidence and smallest repair target; never weaken the gate.

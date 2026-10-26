---
args: { draft: DraftScaffoldingPlan, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ScaffoldingPlan field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Adaptation preserves assessment validity and does not substitute generated work for learner evidence. Explain the decisive evidence and smallest repair target; never weaken the gate.

---
args: { draft: DraftReuseOptionReview, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ReuseOptionReview field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No unqualified feasibility or compliance claim beyond available professional evidence. Explain the decisive evidence and smallest repair target; never weaken the gate.

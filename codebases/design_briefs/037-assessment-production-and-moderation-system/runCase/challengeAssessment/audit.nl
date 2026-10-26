---
args: { draft: DraftItemChallenge, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ItemChallenge field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No unresolved correctness or material ambiguity defect in released items. Explain the decisive evidence and smallest repair target; never weaken the gate.

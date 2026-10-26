---
args: { draft: DraftChangePackage, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ChangePackage field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No unapproved scope or schedule change silently enters the committed baseline. Explain the decisive evidence and smallest repair target; never weaken the gate.

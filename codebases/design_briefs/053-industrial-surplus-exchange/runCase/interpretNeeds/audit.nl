---
args: { draft: DraftReuseNeed, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ReuseNeed field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Recipient confirms constraints that determine suitability or safety. Explain the decisive evidence and smallest repair target; never weaken the gate.

---
args: { draft: DraftMisconceptionHypothesis, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated MisconceptionHypothesis field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No diagnosis based on identity or a single uninformative response. Explain the decisive evidence and smallest repair target; never weaken the gate.

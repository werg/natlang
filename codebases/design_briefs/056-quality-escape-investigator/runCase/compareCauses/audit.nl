---
args: { draft: DraftQualityHypothesis, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated QualityHypothesis field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Causal claims remain provisional until supported by discriminating evidence. Explain the decisive evidence and smallest repair target; never weaken the gate.

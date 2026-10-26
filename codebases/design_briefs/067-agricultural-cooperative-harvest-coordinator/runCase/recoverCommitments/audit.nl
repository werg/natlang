---
args: { draft: DraftHarvestCommitment, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated HarvestCommitment field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every critical quality and substitution condition is explicit. Explain the decisive evidence and smallest repair target; never weaken the gate.

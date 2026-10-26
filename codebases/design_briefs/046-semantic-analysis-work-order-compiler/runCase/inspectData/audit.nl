---
args: { draft: DraftDataSuitability, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DataSuitability field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Each required concept has a supported mapping or an explicit unanswerable component. Explain the decisive evidence and smallest repair target; never weaken the gate.

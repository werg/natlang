---
args: { draft: DraftAssetObservation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated AssetObservation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Observations do not become established faults without supporting evidence. Explain the decisive evidence and smallest repair target; never weaken the gate.

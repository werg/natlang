---
args: { draft: DraftAssetIdentity, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated AssetIdentity field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Evidence attaches to the correct asset instance and revision. Explain the decisive evidence and smallest repair target; never weaken the gate.

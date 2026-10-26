---
args: { draft: DraftHarvestMatch, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated HarvestMatch field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Allocated lots satisfy confirmed essential requirements. Explain the decisive evidence and smallest repair target; never weaken the gate.

---
args: { draft: DraftStockObservation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated StockObservation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Observations do not directly overwrite authoritative stock identity or quantity. Explain the decisive evidence and smallest repair target; never weaken the gate.

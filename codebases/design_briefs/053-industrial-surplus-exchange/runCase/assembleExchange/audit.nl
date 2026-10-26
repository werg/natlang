---
args: { draft: DraftExchangePackage, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ExchangePackage field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No double allocation, unsupported capacity, or negative quantity. Explain the decisive evidence and smallest repair target; never weaken the gate.

---
args: { draft: DraftShippingRecovery, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ShippingRecovery field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Actual carrier state and customer-approved commitments agree before closure. Explain the decisive evidence and smallest repair target; never weaken the gate.

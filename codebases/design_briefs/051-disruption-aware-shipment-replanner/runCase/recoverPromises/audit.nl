---
args: { draft: DraftShippingCommitment, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ShippingCommitment field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every priority follows the supplied policy and actual commitment evidence. Explain the decisive evidence and smallest repair target; never weaken the gate.

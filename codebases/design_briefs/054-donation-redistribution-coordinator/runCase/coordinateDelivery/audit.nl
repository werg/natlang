---
args: { draft: DraftDonationDelivery, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DonationDelivery field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Delivery and acceptance are distinct; recipient details are shared only as needed. Explain the decisive evidence and smallest repair target; never weaken the gate.

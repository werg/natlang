---
args: { draft: DraftDonationOffer, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DonationOffer field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Only confirmed quantities enter allocatable stock. Explain the decisive evidence and smallest repair target; never weaken the gate.

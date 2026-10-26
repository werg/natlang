---
args: { draft: DraftDonationAllocation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DonationAllocation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact conservation and consistent application of the approved allocation policy. Explain the decisive evidence and smallest repair target; never weaken the gate.

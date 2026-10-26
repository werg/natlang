---
args: { draft: DraftDonationBundle, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DonationBundle field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Bundle is usable under the recipient’s stated constraints. Explain the decisive evidence and smallest repair target; never weaken the gate.

---
args: { draft: DraftBidScope, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated BidScope field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every scope statement retains its original qualification and bid version. Explain the decisive evidence and smallest repair target; never weaken the gate.

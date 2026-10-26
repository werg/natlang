---
args: { draft: DraftReuseReceipt, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ReuseReceipt field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Transfer and acceptance evidence support reported quantities and outcomes. Explain the decisive evidence and smallest repair target; never weaken the gate.

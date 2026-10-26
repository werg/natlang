---
args: { draft: DraftDebtOutcome, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DebtOutcome field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Completion requires evidence of the intended operational effect, not only task closure. Explain the decisive evidence and smallest repair target; never weaken the gate.

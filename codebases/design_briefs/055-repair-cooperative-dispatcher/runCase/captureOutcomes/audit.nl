---
args: { draft: DraftRepairOutcome, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated RepairOutcome field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Outcome is verified and lessons retain applicability limits. Explain the decisive evidence and smallest repair target; never weaken the gate.

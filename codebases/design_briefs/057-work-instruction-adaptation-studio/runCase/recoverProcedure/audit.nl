---
args: { draft: DraftProcedureRequirement, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ProcedureRequirement field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every mandatory control has a traceable source and cannot disappear during simplification. Explain the decisive evidence and smallest repair target; never weaken the gate.

---
args: { draft: DraftChangeOption, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ChangeOption field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Option is technically reviewable, commercially explicit, and dependency-feasible. Explain the decisive evidence and smallest repair target; never weaken the gate.

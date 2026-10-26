---
args: { draft: DraftReuseMatch, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ReuseMatch field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Essential fit is evidenced; speculative compatibility creates a qualification task, not an allocation. Explain the decisive evidence and smallest repair target; never weaken the gate.

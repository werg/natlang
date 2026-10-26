---
args: { draft: DraftHandoverClosureTask, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated HandoverClosureTask field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every mandatory gap has an accountable closure path or explicit escalation. Explain the decisive evidence and smallest repair target; never weaken the gate.

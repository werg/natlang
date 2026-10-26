---
args: { draft: DraftPracticeTransition, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated PracticeTransition field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Resource, knowledge, and procedure invariants remain valid after every transition. Explain the decisive evidence and smallest repair target; never weaken the gate.

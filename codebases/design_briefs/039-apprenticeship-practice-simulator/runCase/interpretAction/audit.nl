---
args: { draft: DraftLearnerAction, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated LearnerAction field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Action maps to a permitted typed operation or remains unexecuted. Explain the decisive evidence and smallest repair target; never weaken the gate.

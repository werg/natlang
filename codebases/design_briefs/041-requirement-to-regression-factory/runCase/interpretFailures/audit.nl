---
args: { draft: DraftTestFailureFinding, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated TestFailureFinding field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Failure classification is evidence-backed and environment errors are not counted as product coverage. Explain the decisive evidence and smallest repair target; never weaken the gate.

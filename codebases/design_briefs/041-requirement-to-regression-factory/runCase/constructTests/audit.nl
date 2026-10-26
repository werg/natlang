---
args: { draft: DraftRegressionTest, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated RegressionTest field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Tests compile and have independent expected behavior, not output copied from the implementation. Explain the decisive evidence and smallest repair target; never weaken the gate.

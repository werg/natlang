---
args: { draft: DraftCompatibilityTest, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated CompatibilityTest field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Paired replay uses identical inputs and independent consumer acceptance criteria. Explain the decisive evidence and smallest repair target; never weaken the gate.

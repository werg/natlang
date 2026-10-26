---
args: { draft: DraftChangeWorkPackage, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ChangeWorkPackage field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Feasible version transition and explicit handling of existing stock and in-flight work. Explain the decisive evidence and smallest repair target; never weaken the gate.

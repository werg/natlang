---
args: { draft: DraftProjectConcept, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ProjectConcept field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: All roles contribute substantive evidence and the challenge is solvable with available resources. Explain the decisive evidence and smallest repair target; never weaken the gate.

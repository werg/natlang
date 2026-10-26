---
args: { draft: DraftProjectMilestone, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ProjectMilestone field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Acyclic dependency graph, feasible workload, and complete competency coverage. Explain the decisive evidence and smallest repair target; never weaken the gate.

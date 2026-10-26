---
args: { draft: DraftCompetencyMapping, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated CompetencyMapping field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every mapping names the demonstrated behavior and source location. Explain the decisive evidence and smallest repair target; never weaken the gate.

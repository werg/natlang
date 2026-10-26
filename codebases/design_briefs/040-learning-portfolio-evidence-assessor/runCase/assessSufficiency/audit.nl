---
args: { draft: DraftCompetencyFinding, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated CompetencyFinding field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Sufficiency follows the approved standard and does not rely on unsupported confidence. Explain the decisive evidence and smallest repair target; never weaken the gate.

---
args: { draft: DraftIncidentImpact, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated IncidentImpact field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Impact statements cite observations and preserve unknown coverage. Explain the decisive evidence and smallest repair target; never weaken the gate.

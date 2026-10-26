---
args: { draft: DraftIncidentHypothesis, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated IncidentHypothesis field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Hypotheses remain provisional until diagnostic evidence supports them. Explain the decisive evidence and smallest repair target; never weaken the gate.

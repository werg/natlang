---
args: { draft: DraftEnvironmentalSubmission, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated EnvironmentalSubmission field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Narrative, calculations, methods, and approved boundary all agree. Explain the decisive evidence and smallest repair target; never weaken the gate.

---
args: { draft: DraftHandoverDossier, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated HandoverDossier field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No claim of technical readiness based solely on a complete document bundle. Explain the decisive evidence and smallest repair target; never weaken the gate.

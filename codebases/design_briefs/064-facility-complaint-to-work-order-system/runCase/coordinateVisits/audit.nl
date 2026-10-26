---
args: { draft: DraftFacilityVisit, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated FacilityVisit field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No unauthorized access or impossible trade sequence. Explain the decisive evidence and smallest repair target; never weaken the gate.

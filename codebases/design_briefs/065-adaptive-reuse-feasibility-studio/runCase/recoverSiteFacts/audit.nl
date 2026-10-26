---
args: { draft: DraftReuseSiteFact, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ReuseSiteFact field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No assumed structural or regulatory fact without supplied evidence. Explain the decisive evidence and smallest repair target; never weaken the gate.

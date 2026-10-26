---
args: { draft: DraftSurplusOffer, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated SurplusOffer field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No unsupported material property or transferable-title claim. Explain the decisive evidence and smallest repair target; never weaken the gate.

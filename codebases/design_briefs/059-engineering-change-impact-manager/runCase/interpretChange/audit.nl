---
args: { draft: DraftDesignDelta, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DesignDelta field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every effect has a design-source basis and explicit scope. Explain the decisive evidence and smallest repair target; never weaken the gate.

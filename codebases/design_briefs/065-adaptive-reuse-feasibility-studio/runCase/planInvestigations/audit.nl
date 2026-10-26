---
args: { draft: DraftReuseInvestigation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ReuseInvestigation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Each investigation resolves a named uncertainty and uses qualified professionals where required. Explain the decisive evidence and smallest repair target; never weaken the gate.

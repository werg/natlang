---
args: { draft: DraftPortfolioEvidence, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated PortfolioEvidence field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Evidence type and authorship status are explicit. Explain the decisive evidence and smallest repair target; never weaken the gate.

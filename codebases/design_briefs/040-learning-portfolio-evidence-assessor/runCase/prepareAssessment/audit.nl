---
args: { draft: DraftPortfolioDisposition, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated PortfolioDisposition field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: High-stakes outcomes require accountable assessor approval and an appeal path. Explain the decisive evidence and smallest repair target; never weaken the gate.

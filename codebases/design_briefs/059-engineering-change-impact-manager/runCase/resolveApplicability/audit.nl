---
args: { draft: DraftChangeApplicability, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ChangeApplicability field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact applicability rules are established before downstream propagation. Explain the decisive evidence and smallest repair target; never weaken the gate.

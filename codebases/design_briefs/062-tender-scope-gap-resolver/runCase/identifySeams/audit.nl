---
args: { draft: DraftScopeInterface, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ScopeInterface field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every required interface has an accepted owner or an explicit unresolved gap. Explain the decisive evidence and smallest repair target; never weaken the gate.

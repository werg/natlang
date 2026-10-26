---
args: { draft: DraftIdentityCluster, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated IdentityCluster field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No cluster contains an established cannot-link pair or impossible identity chronology. Explain the decisive evidence and smallest repair target; never weaken the gate.

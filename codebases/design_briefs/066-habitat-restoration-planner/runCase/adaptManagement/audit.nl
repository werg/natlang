---
args: { draft: DraftRestorationRevision, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated RestorationRevision field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Adaptive changes retain evidence, uncertainty, and current access or intervention authority. Explain the decisive evidence and smallest repair target; never weaken the gate.

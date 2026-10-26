---
args: { draft: DraftConsumerDependency, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ConsumerDependency field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Dependency claims are specific to behavior, not mere package-name overlap. Explain the decisive evidence and smallest repair target; never weaken the gate.

---
args: { draft: DraftDatasetUse, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DatasetUse field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every required slice and restriction has a use-based rationale. Explain the decisive evidence and smallest repair target; never weaken the gate.

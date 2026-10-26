---
args: { draft: DraftDisruptionImpact, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DisruptionImpact field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Scope is evidence-backed and uncertain reopening times are not treated as confirmed capacity. Explain the decisive evidence and smallest repair target; never weaken the gate.

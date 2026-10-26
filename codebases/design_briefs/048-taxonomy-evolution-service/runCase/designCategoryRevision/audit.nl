---
args: { draft: DraftCategoryRevision, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated CategoryRevision field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No cycle, contradictory boundary, or silent reuse of an ID for a different concept. Explain the decisive evidence and smallest repair target; never weaken the gate.

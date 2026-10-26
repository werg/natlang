---
args: { draft: DraftSourceFieldMeaning, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated SourceFieldMeaning field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No target mapping based solely on field names or observed value formats. Explain the decisive evidence and smallest repair target; never weaken the gate.

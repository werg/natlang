---
args: { draft: DraftAnalysisMeaning, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated AnalysisMeaning field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every phrase affecting inclusion, timing, or aggregation maps to an explicit meaning component. Explain the decisive evidence and smallest repair target; never weaken the gate.

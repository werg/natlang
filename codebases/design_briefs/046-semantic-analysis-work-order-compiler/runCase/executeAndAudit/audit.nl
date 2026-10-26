---
args: { draft: DraftAnalysisSlice, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated AnalysisSlice field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No silent row loss, duplicate join inflation, unclassified inclusion, or unsupported denominator. Explain the decisive evidence and smallest repair target; never weaken the gate.

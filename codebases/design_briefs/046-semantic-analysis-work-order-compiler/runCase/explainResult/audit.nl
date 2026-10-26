---
args: { draft: DraftAnalysisAnswer, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated AnalysisAnswer field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every numerical claim matches the materialized result and every semantic qualification matches the plan. Explain the decisive evidence and smallest repair target; never weaken the gate.

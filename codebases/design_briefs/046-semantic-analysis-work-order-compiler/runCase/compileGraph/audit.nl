---
args: { draft: DraftAnalysisWorkGraph, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated AnalysisWorkGraph field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Graph typechecks, has no cycle, and preserves the formalized population and output grain. Explain the decisive evidence and smallest repair target; never weaken the gate.

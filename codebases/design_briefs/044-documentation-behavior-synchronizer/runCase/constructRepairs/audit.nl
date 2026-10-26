---
args: { draft: DraftDocumentationPatch, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DocumentationPatch field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact patch bases and semantic preservation checks pass. Explain the decisive evidence and smallest repair target; never weaken the gate.

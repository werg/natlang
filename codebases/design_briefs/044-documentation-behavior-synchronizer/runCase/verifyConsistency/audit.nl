---
args: { draft: DraftDocumentationRelease, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DocumentationRelease field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No unresolved blocking mismatch or broken executable example in the released bundle. Explain the decisive evidence and smallest repair target; never weaken the gate.

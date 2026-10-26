---
args: { draft: DraftDocumentationMismatch, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DocumentationMismatch field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Repair direction has authoritative support rather than convenience. Explain the decisive evidence and smallest repair target; never weaken the gate.

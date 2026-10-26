---
args: { draft: DraftDocumentedPromise, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DocumentedPromise field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every promise has an applicable version and source span. Explain the decisive evidence and smallest repair target; never weaken the gate.

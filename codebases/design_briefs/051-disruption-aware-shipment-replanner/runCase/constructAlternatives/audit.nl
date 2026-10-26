---
args: { draft: DraftShippingAlternative, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ShippingAlternative field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Essential constraints remain intact; changed promises require explicit customer or owner approval. Explain the decisive evidence and smallest repair target; never weaken the gate.

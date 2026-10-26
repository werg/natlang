---
args: { draft: DraftWarehouseFinding, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated WarehouseFinding field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact affected quantities and evidence-backed exception categories. Explain the decisive evidence and smallest repair target; never weaken the gate.

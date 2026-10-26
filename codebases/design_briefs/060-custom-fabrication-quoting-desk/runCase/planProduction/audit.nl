---
args: { draft: DraftFabricationRoute, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated FabricationRoute field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact resource and process feasibility; no unsupported engineering certification. Explain the decisive evidence and smallest repair target; never weaken the gate.

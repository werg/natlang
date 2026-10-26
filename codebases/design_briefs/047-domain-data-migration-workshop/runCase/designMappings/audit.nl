---
args: { draft: DraftMigrationMapping, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated MigrationMapping field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every mapping declares its domain, missing-value behavior, and information loss. Explain the decisive evidence and smallest repair target; never weaken the gate.

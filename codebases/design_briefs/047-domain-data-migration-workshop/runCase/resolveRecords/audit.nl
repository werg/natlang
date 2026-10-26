---
args: { draft: DraftMigrationDecision, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated MigrationDecision field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Accepted target records satisfy schema and source-meaning checks. Explain the decisive evidence and smallest repair target; never weaken the gate.

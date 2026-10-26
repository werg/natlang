---
args: { draft: DraftMigrationRehearsal, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated MigrationRehearsal field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact counts, relationships, conservation constraints, and reversible mappings verified. Explain the decisive evidence and smallest repair target; never weaken the gate.

---
args: { draft: DraftMigrationWave, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated MigrationWave field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Each wave is feasible and has a tested rollback or explicit irreversible-change approval. Explain the decisive evidence and smallest repair target; never weaken the gate.

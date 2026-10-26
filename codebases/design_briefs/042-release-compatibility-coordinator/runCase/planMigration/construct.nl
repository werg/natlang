---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftMigrationWave
---
Construct a MigrationWave draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Construct staged migration plans with prerequisites, owners, compatibility shims where approved, and rollback conditions. Group consumers by meaningful risk and avoid advancing a wave before its required operational readiness evidence exists.

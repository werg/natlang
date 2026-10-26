---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftMigrationCutover
---
Construct a MigrationCutover draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Freeze the approved mapping and cutover scope, apply final source deltas, and reconcile target results before switching consumers. Track quarantined records and post-cutover corrections explicitly, retaining a safe rollback or forward-repair plan for irreversible changes.

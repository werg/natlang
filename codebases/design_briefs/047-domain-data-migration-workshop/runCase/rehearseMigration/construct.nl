---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftMigrationRehearsal
---
Construct a MigrationRehearsal draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Materialize a staged target snapshot, validate referential integrity and business invariants, and compare representative source-target round trips. Rehearse rollback and delta capture so cutover does not lose changes made after the initial extraction.

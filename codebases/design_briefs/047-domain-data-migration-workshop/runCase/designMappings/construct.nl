---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftMigrationMapping
---
Construct a MigrationMapping draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Construct conditional field and entity mappings that preserve domain meaning. Identify lossy transformations, required target extensions, and records needing manual resolution; do not invent missing facts to satisfy a stricter target schema.

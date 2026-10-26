---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDatasetRelease
---
Construct a DatasetRelease draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Create targeted acquisition or annotation tasks, rerun audits on new material, and assemble a release manifest with known limits. Freeze splits by source and world family, prevent leakage, and require owner approval for the specific release use.

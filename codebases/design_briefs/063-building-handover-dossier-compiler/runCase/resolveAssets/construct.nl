---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftAssetIdentity
---
Construct a AssetIdentity draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile asset tags, drawing labels, manuals, and contractor naming conventions. Distinguish identical models at different locations, preserve serial and revision data, and flag ambiguous identity before attaching a certificate or manual to the wrong asset.

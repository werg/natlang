---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftHandoverEvidence
---
Construct a HandoverEvidence draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare submitted manuals, inspection records, certificates, and training logs with the actual requirement. Distinguish a blank form from completed evidence and a generic manual from the installed configuration; retain missing signatures or scope mismatches.

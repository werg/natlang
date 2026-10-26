---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftRestorationRevision
---
Construct a RestorationRevision draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Interpret new observations against baseline and monitoring rules, including alternative explanations such as seasonality or survey changes. Propose bounded plan revisions, preserve unaffected work, and obtain expert approval before changing interventions.

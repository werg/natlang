---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftCategoryRevision
---
Construct a CategoryRevision draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Draft revised definitions, inclusion and exclusion rules, and parent-child relationships. Preserve stable category IDs where meaning remains stable, create new identities when meaning changes, and explicitly map split or merged historical categories.

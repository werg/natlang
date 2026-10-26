---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDocumentationMismatch
---
Construct a DocumentationMismatch draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile requirements, release decisions, tests, and observed behavior to determine which artifact is wrong. Do not automatically rewrite documentation to match a bug; preserve unresolved intent questions for the responsible product or engineering owner.

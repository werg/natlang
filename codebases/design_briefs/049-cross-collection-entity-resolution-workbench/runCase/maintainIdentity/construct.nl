---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftIdentityRevision
---
Construct a IdentityRevision draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Process new evidence and reviewer corrections as reversible identity changes. Trace downstream uses, prepare split or merge migrations, and preserve old identifiers as redirects with explicit validity rather than silently changing historical joins.

---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDocumentationPatch
---
Construct a DocumentationPatch draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Draft coordinated changes to pages, examples, tests, and release notes according to the reviewed repair direction. Preserve unaffected claims, update cross-links and terminology, and keep code examples executable under their stated environment.

---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDocumentationRelease
---
Construct a DocumentationRelease draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Re-extract changed promises, rerun examples and targeted behavior probes, and inspect linked pages for stale repetitions. Publish the reviewed bundle only when its examples and claims correspond to the same release version.

---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDocumentedPromise
---
Construct a DocumentedPromise draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Extract checkable statements about inputs, outputs, defaults, errors, and supported workflows from documentation and examples. Preserve version and audience scope, distinguishing normative promises from illustrative suggestions or historical notes.

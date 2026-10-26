---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftAnnotationRule
---
Construct a AnnotationRule draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Draft operational labeling instructions with examples, counterexamples, and uncertainty handling. Separate observable labels from inferred attributes, define adjudication, and test whether independent annotators can apply the rule consistently.

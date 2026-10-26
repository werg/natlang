---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftIncidentHypothesis
---
Construct a IncidentHypothesis draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Generate a bounded set of causal hypotheses from deployment history, dependencies, and symptoms. Track evidence for and against each, choose checks that distinguish mechanisms, and avoid repeatedly confirming the most recent or most fluent explanation.

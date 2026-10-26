---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftAnalysisMeaning
---
Construct a AnalysisMeaning draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Parse the request into population, entities, semantic predicates, temporal relations, grouping, and desired output. Generate competing interpretations only where they change the answer; resolve decisive ambiguity before choosing a convenient available column.

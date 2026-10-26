---
args: { item: AssessmentItemDraft, competencies: CompetencyTrace[], assessment: AssessmentBuildRequest }
returns: IndependentSolutionSet
---
Solve the item from its prompt and allowed resources without consulting its proposed key. Record each distinct method that satisfies the requested evidence, including methods that use different intermediate forms. Mark any answer that depends on unstated information. Return the actual worked route descriptions and the conclusion for each material reading; do not copy a model answer into the independent-solve field.

---
args: { items: AssessmentItemDraft[], competencies: CompetencyTrace[], assessment: AssessmentBuildRequest }
returns: RubricSpec[]
---
1. Bind criteria = defineVisibleScoringCriteria(items, competencies).
2. Bind alternatives = enumerateValidMethods(items, assessment). Treat conceptual correctness separately from presentation preference.
3. Bind partialCredit = placePartialCreditBoundaries(criteria, alternatives).
4. For ambiguous or unfamiliar responses bind review = preserveSemanticReviewPath(alternatives, partialCredit). Do not auto-reject work that is valid but unlike the model answer.

---
args: { responseMethod: string, rubric: RubricSpec, item: AssessmentItemDraft }
returns: MethodDisposition
---
Compare the response's reasoning steps with each rubric criterion and every accepted method already named for this item. If its route differs but establishes the same competency evidence, preserve the response as a valid method candidate and send it to semantic review; do not reject it because the model answer uses another route. If the route changes the competency being assessed, name that mismatch and leave the score unresolved. Return the supporting criterion references and the exact unresolved proposition.

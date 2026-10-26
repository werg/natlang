---
args: { item: AssessmentItemDraft, rubric: RubricSpec, solutions: IndependentSolutionSet }
returns: BorderlineReplay
---
Construct borderline response classes from the independently solved methods: complete reasoning, correct result with missing explanation, arithmetic slip after a valid setup, and a plausible response under each material alternate reading. Apply each visible rubric criterion separately. Flag whenever equivalent competency evidence receives different treatment or a response is penalized for incidental presentation. Return the response class, criterion trace, and disagreement location.

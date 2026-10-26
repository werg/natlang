---
description: Split each consultation submission into attributable, scoped arguments without flattening mixed positions.
args: { docket: ConsultationDocket; delta: SubmissionDelta }
returns: ScopedArgument[]
---
For each new submission, retain each separately expressed position, its reason, factual claims, requested action and affected proposal scope. One submission may yield several arguments with different positions. Do not infer a single positive/negative label or create motives not stated. Keep late or incomplete submissions visible with their exact inclusion basis and source references.

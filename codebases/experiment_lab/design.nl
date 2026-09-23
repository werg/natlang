---
description: Select a bounded, informative set of predeclared scenarios for an experiment.
args:
  question: string
  cases: Case[]
  budget: number
returns: Plan
---
Choose at most `budget` distinct case IDs from `cases` that best probe
`question`. Cover distinct groups and include a challenging case when one
is offered. Return only listed IDs and a short reason. The host will validate
the selection and hold all candidate inputs fixed before any trial starts. Do not
claim that selecting a case has tested it.
Write exactly a `Plan` record with `selected: string[]` and `reason: string`.
The selected field contains the IDs. Do not use an `ids` field.

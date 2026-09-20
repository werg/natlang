---
description: Select a bounded, informative set of predeclared scenarios for an experiment.
args:
  question: Text
  cases: Case[]
  budget: Num
returns: Plan
types:
  Case: '{ id: Text, group: Text, description: Text, expected: Text }'
  Plan: '{ selected: Text[], reason: Text }'
---
Choose at most `args/budget` distinct case IDs from `args/cases` that best probe
`args/question`. Cover distinct groups and include a challenging case when one
is offered. Return only listed IDs and a short reason. The host will validate
the selection and hold all candidate inputs fixed before any trial starts. Do not
claim that selecting a case has tested it.
Write exactly a `Plan` record with `selected: Text[]` and `reason: Text`.
The selected field contains the IDs. Do not use an `ids` field.

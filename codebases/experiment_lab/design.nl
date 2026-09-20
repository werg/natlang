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
`args/question`. Include an informative conflict case when the question concerns
semantic merging. Return only listed IDs and a short reason. The host will validate
the selection and hold all candidate inputs fixed before any trial starts. Do not
claim that selecting a case has tested it.

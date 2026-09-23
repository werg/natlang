---
description: Assess each observation and retain its reason alongside exact counts.
args:
  observations: string[]
  criterion: string
returns: Report
---
Assess every observation against the criterion with assess; the assessments are
independent, so they can run together. Then return summarize of the assessments,
in the original order.

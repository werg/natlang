---
description: Label one ticket using the rubric.
args:
  ticket: string
  rubric: string
returns: Label
---
Pick the label from `args/rubric` that fits `args/ticket`.
If no rule in the rubric covers the ticket, report a blocker saying what is missing.

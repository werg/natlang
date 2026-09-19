---
description: Triage a batch of support tickets into a report.
args:
  tickets: Text[]
  rubric: Text
returns: Report
types:
  Label: '"billing" | "technical" | "spam"'
  Report: '{ urgent: Num, by_label: Dict<Num>, summary: Text }'
uses:
  count_true: ../std/count_true
  select_by_flags: ../std/select_by_flags
  group_count: ../std/group_count
---
function triage(tickets, rubric) -> Report

  labels   = for each t in tickets: classify(t, rubric)
  not_spam = for each l in labels: l is not "spam"            # exact: use code
  real     = select_by_flags(tickets, not_spam)
  flags    = for each t in real: is_urgent(t)
  urgent   = select_by_flags(real, flags)

  if urgent is empty:
      summary = "Nothing urgent today."
  else:
      summary = summarize(urgent)

  return { urgent: count_true(flags), by_label: group_count(labels), summary }

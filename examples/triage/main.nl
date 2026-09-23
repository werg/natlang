import classify from "./main/classify";
import is_urgent from "./main/is_urgent";
import summarize from "./main/summarize";
import count_true from "../std/count_true";
import select_by_flags from "../std/select_by_flags";
import group_count from "../std/group_count";
---
description: Triage a batch of support tickets into a report.
args:
  tickets: string[]
  rubric: string
returns: Report
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

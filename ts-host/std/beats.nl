---
description: Decide whether the first of two candidates serves a goal better than the second.
args:
  goal: string
  first: unknown
  second: unknown
returns: boolean
readout: decision
---
first and second are two candidates for goal. Answer true when first serves goal better than second, false when
second serves it at least as well. Judge what each candidate does for goal, not its length or wording.

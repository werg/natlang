---
description: Check that an edit does what the agent said it would.
readout: decision
args:
  intent: string
  diff: string
returns: EditCheck
---
A coding agent edited a file; intent is what it said it was about to do, and diff is the change. as-intended: the
diff does that and nothing else. unintended: it also deletes, reverts or changes code the intent does not cover.
incomplete: it does only part of what intent says, or leaves the code inconsistent (a renamed name still used, an
unclosed block).

---
description: Whether a legacy program is worth migrating into the new style or should be deleted.
args:
  brief: string
  baseline: Baseline
  findings: Finding[]
returns: Keep
readout: decision
---
brief says what the program is for. baseline is how it did on its own exact cases on the local model before any
change, and findings are the places that still use the old style. Answer "migrate" when the program does real work
on its cases (some pass) and the findings are places a rewrite can fix. Answer "delete" when the program teaches
nothing its cases can confirm: its cases almost never pass, or what it does is only plumbing written as prose.

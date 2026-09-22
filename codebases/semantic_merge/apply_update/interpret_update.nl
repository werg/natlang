---
description: Semantically merge one incoming update with a prior semantic state and its unresolved alternatives.
args:
  base: Document
  current: MergeResult
  update: Update
  policy: string
returns: Draft
---
Consider `args/current/text`, every prior applied update and every unresolved alternative.
Interpret the new update's intent under `args/policy` and the agreed `args/base`. Return a
new proposed document, the IDs whose intentions it now represents, and alternatives for
all remaining unresolved IDs. Account for every prior update and the new update exactly
once. Do not discard an old ambiguity merely because this step received a new event.
This is an incremental semantic decision; it need not equal a fresh whole-history merge.

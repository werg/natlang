---
description: Decide whether two edits of the same value can both be kept.
args:
  base: unknown
  ours: Edit
  theirs: Edit
returns: boolean
readout: decision
types:
  Edit: { after: unknown, intent: string }
---
ours and theirs are two separate edits of base, each with the value it produced and what it is for. Answer true when
one value can carry out both purposes at once. Answer false when keeping one purpose undoes, contradicts or replaces
the other.

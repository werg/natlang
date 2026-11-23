---
description: Decide whether two edits of the same value can both be kept.
args:
  base?: unknown
  ours: Edit
  theirs: Edit
returns: boolean
readout: decision
types:
  Edit: "{ after?: unknown, removed?: boolean, intent: string }"
---
ours and theirs are two separate edits of base (an omitted base means both sides added the field; removed: true means
that side deleted it), each with the value it produced and what it is for. Answer true when
one value can carry out both purposes at once. Answer false when keeping one purpose undoes, contradicts or replaces
the other.

---
description: Write one value that keeps two compatible edits of the same base.
args:
  base?: unknown
  ours: Edit
  theirs: Edit
returns: unknown
types:
  Edit: "{ after?: unknown, removed?: boolean, intent: string }"
---
ours and theirs are two edits of base (an omitted base means both sides added the field; removed: true means that side
deleted it), each with the value it produced and what it is for. Write the value that makes both edits: start from base
when present, carry out the purpose of ours and the purpose of theirs, and change nothing else. Keep the same kind of
value and style as the edits.

---
description: Write one value that keeps two compatible edits of the same base.
args:
  base: unknown
  ours: Edit
  theirs: Edit
returns: unknown
types:
  Edit: { after: unknown, intent: string }
---
ours and theirs are two edits of base, each with the value it produced and what it is for. Write the value that makes
both edits: start from base, carry out the purpose of ours and the purpose of theirs, and change nothing else. The
result is the same kind of value as base, written in the same style.

---
description: Semantically reconcile scene object changes.
args:
  base: State
  updates: Update[]
  policy: Text
returns: Draft
---
Interpret object references and intent in the scene. Two authors may move one object,
recolor it independently, create nearby but distinct objects, or delete an object while
another edits it. Keep object IDs stable when identity is clear. Do not infer identity
from proximity alone. Return typed objects, applied update IDs and unresolved alternatives
for cases that `policy` cannot settle.

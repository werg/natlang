---
description: Semantically reconcile an agreed base and a finite set of updates, preserving unresolved alternatives.
args:
  base: Document
  updates: Update[]
  policy: Text
returns: MergeResult
---
function merge_history(base, updates, policy) -> MergeResult
  prepared = prepare(base, updates)
  if not prepared.valid:
    return reject(base, prepared)
  if prepared.updates is empty:
    return unchanged(base, prepared)
  draft = interpret_history(base, prepared.updates, policy)
  return finish(base, prepared, draft)

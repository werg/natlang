---
description: Incrementally merge one semantic update into a prior result, retaining provenance and unresolved intent.
args:
  base: Document
  current: MergeResult
  update: Update
  policy: string
returns: MergeResult
---
function apply_update(base, current, update, policy) -> MergeResult
  step = prepare_step(base, current, update)
  if step.kind is "duplicate": return current
  if step.kind is "invalid": return reject_step(current, update, step)
  draft = interpret_update(base, current, update, policy)
  return finish_step(current, update, step, draft)

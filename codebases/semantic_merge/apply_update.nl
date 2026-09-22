import { finish_step } from "./apply_update/finish_step";
import { interpret_update } from "./apply_update/interpret_update";
import { prepare_step } from "./apply_update/prepare_step";
import { reject_step } from "./apply_update/reject_step";
---
description: Incrementally merge one semantic update into a prior result, retaining provenance and unresolved intent.
args:
  base: Document
  current: MergeResult
  update: Update
  policy: Text
returns: MergeResult
---
function apply_update(base, current, update, policy) -> MergeResult
  step = prepare_step(base, current, update)
  if step.kind is "duplicate": return current
  if step.kind is "invalid": return reject_step(current, update, step)
  draft = interpret_update(base, current, update, policy)
  return finish_step(current, update, step, draft)

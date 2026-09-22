import { finish } from "./merge_history/finish";
import { interpret_history } from "./merge_history/interpret_history";
import { prepare } from "./merge_history/prepare";
import { reject } from "./merge_history/reject";
import { unchanged } from "./merge_history/unchanged";
---
description: Semantically reconcile an agreed base and a finite set of updates, preserving unresolved alternatives.
args:
  base: Document
  updates: Update[]
  policy: string
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

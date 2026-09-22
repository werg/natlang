---
description: Semantically reconcile ordered items.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
---
Interpret item identity, order and wording together. A move is not a delete plus an
unrelated insertion when the author means to retain the item. Preserve stable item IDs.
Consider concurrent inserts at the same location, edit versus delete, split versus rename,
and dependencies between instructions. Return the typed list and an account for every
update ID. If the intended order cannot be resolved under `policy`, retain alternatives.

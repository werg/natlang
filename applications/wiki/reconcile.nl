---
description: Reconcile concurrent edits to a wiki page's stable blocks.
args:
  base: WikiPage
  updates: WikiUpdate[]
returns: MergeDraft
---
Reconcile the meaning of the updates to each stable block of base. All updates
have the same base revision and are presented in stable ID order. Preserve
compatible changes, and record an unresolved conflict with its alternatives when
meanings conflict. Account for every update ID exactly once in accounted. Keep
the original block IDs and order. For a code cell, return runnable source only if
the combined meaning is clear; otherwise keep the old source and record a
conflict.

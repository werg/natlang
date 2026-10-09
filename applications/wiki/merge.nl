---
description: The edit pipeline. Reconcile concurrent updates to a wiki page, block by block, into a draft the wiki can publish.
args:
  base: WikiPage
  updates: WikiUpdate[]
  settings: WikiSettings
returns: MergeDraft
---
Reconcile updates, made concurrently against base, a wiki page, into a MergeDraft. All updates have the same base
revision and arrive in update ID order. Blocks are independent, so reconcile each block by itself with block, the stage
in your folder.

1. Group updates by block_id, keeping update ID order within each group.
2. For each block of base that has updates, all at once: block(that block, its group, settings).
3. blocks: the blocks of base in their order, each replaced by the block of its outcome when it has one. The block IDs
   and order are base's.
4. accounted: the accounted lists of all the outcomes, joined. unresolved: the unresolved lists of all the outcomes,
   joined, in block order.

Return { blocks, accounted, unresolved }. Every update ID is in accounted once.

---
description: Merge planning. From the changes to one block and how each pair relates, decide which updates combine, which are covered, and which conflict.
args:
  block_id: string
  changes: Change[]
  relations: Related[]
returns: MergePlan
---
Plan the merge of changes, all made to block block_id from the same base text. relations holds the relation of every
pair a, b of update IDs (a sorts before b). A pair that relations does not list is compatible. Work on update IDs, in
sorted order, with these data structures:

- `conflicting`: a set of update IDs, empty at first.
- `covered`: a map from an update ID to the update ID that covers it, empty at first.

Steps:

1. Conflicts. Take the graph whose nodes are the update IDs and whose edges are the contradictory pairs, and find its
   connected components. Every component of two or more IDs is one conflict: add its IDs to `conflicting`, and make one step { action: "conflict", update_ids: the component's
   IDs sorted, note: which fact they disagree on }.
2. Redundancy. Among the IDs not in `conflicting`, for each redundant pair a, b where neither is covered yet: the
   later ID, b, is covered by a, so set covered[b] = a. If a is itself covered, use what covers a. Make one step
   { action: "covered", update_ids: [b], by: a, note: why a carries b } for each.
3. Combination. The IDs that are neither in `conflicting` nor keys of `covered` are combined, in sorted order. When
   there is at least one, make one step { action: "combine", update_ids: those IDs, note: one sentence on how they fit
   together }. Updates of kind noop are listed among the combined IDs like any other.

Every update ID of changes appears in exactly one step, in update_ids (for covered, the covered ID). Return
{ block_id, steps } with the conflict steps first, then covered, then combine.

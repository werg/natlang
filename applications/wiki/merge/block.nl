---
description: Reconcile the updates to one block. Summarize each, find the conflicts, plan the merge, write the merged text and check it against every intent.
args:
  block: WikiBlock
  updates: WikiUpdate[]
  settings: WikiSettings
returns: BlockOutcome
---
Reconcile updates, all made to block from the same base text, with the stages in your folder. Work through these steps.

1. Change summaries. For each update, all at once: changes(block, update, settings). Keep the results in the order of
   the update IDs.
2. Conflict detection. For each pair a, b of those changes where a's update_id sorts before b's, all at once, find
   the relation:
   - "compatible" when a or b has kind "noop" or "format";
   - "redundant" when a.text equals b.text;
   - otherwise d = decide(relate, a, b). The relation is d.value when d.confidence is at least 0.6 and
     "contradictory" when it is lower.
   Keep { a: a's update_id, b: b's update_id, relation } for every pair.
3. Merge plan. plan(block.id, the changes, the relations).
4. Carried changes. `carried` is the changes whose update IDs are in the plan's "combine" step or in a "covered" step's
   update_ids.
5. Merge and verification against every intent, only when the plan has a "combine" step. Repeat the writing step with
   iterateOn: refine.iterateOn({ text: "", clear: true, lost: null }, block, carried, plan).until(a => !a.clear ||
   (a.lost !== null && a.lost.length === 0)).withLimit({ maxSteps: 3 }). One step writes the text, or writes it again
   when an intent is missing, and checks it against each intent. The attempt is the state it stops on. With no
   "combine" step the attempt is { text: block.text, clear: true, lost: [] }.
6. Failure. The merge failed when the attempt is not clear, when it still has lost intents, or when the iteration
   reached its limit. Then the merged text is block.text and every update ID of `carried` is unresolved.
7. Result. The block is block with its text replaced by the merged text. unresolved has one Conflict
   { update_id, block_id: block.id, alternatives } for each update ID in a "conflict" step or in the failure set, where
   alternatives is the text of every update in that step (or, for the failure set, in the failure set), in update ID
   order. accounted is every update ID of updates, in sorted order, once each. Return { block, accounted, unresolved }.

---
description: One step of writing a block's merged text. Write the text when there is none, or write it again when an intent is missing, then check it against every intent.
args:
  attempt: Attempt
  block: WikiBlock
  changes: Change[]
  plan: MergePlan
returns: Attempt
---
Advance attempt, the merged text of block, one step. changes are the changes that are not in conflict. The caller repeats
this step until no intent is lost. Follow these steps.

1. Choose the changes to write: `combined` is the changes whose update IDs are in plan's "combine" step. `problem` is
   empty at first.
2. When attempt.lost is not null, the text was checked before. Collect `lost`: the update IDs in attempt.lost. When it
   is empty, return attempt unchanged. Otherwise add the changes of those IDs to `combined` (sorted by update ID, each
   once), and let problem be one sentence: "The text does not carry: " and the intent of each lost change, joined with
   "; ".
3. Write the text: prose(block, combined, problem) when block.kind is "prose", code(block, combined, problem) when it
   is "cell". When the answer is not clear, return { text: block.text, clear: false, lost: [] }.
4. Check the new text against every one of changes, all at once: decide(honors, change, block.text, the new text).
5. Return { text: the new text, clear: true, lost: the update IDs whose verdict is "partly" or "lost" }.

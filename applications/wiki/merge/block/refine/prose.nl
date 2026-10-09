---
description: Merge. Write the text of a prose block that carries several compatible changes.
args:
  block: WikiBlock
  changes: Change[]
  problem?: string
returns: Merged
---
Write the new text of block, a prose block, so that it carries every one of changes. They were made to the same base
text, block.text, and they are compatible. Write it in these steps.

1. Start from block.text, in its own wording, order and markup (headings, lists, [[links]]).
2. Go through changes in order of update_id. For each, put its adds into the text, at the place its update text puts
   them, and take out its removes. When two changes work on the same sentence, write one sentence that holds the adds
   of both. A change of kind noop or format adds nothing and removes nothing.
3. Leave every sentence that no change touches as it is.
4. Read the result against each change's intent and make sure every intent holds.

Return { text, clear: true, reason: "" }. Return clear false, with the text of block and the reason, only when the
changes do not fit into one text.

problem, when given, says what an earlier text failed to carry; write a text that carries it.

---
description: Change summary. Say what one update does to the block it edits, as a Change.
args:
  block: WikiBlock
  update: WikiUpdate
returns: Change
---
Summarize update against block, the block it edits as the author saw it. Work out the Change in these steps.

1. Compare the two texts. Set kind:
   - noop: the texts are identical;
   - format: the same words and meaning, only spacing, line breaks or markup differ;
   - extend: all of block's text is kept and more is added;
   - trim: all of update's text is in block and some of block is dropped;
   - rewrite: anything else.
2. List adds: each statement, step or behaviour that update's text has and block's text does not, one short phrase
   each. For a cell, a behaviour is something the code does or returns; for prose, a claim or an instruction.
3. List removes: each statement, step or behaviour that block's text has and update's text drops or reverses, one
   short phrase each.
4. Write intent: one sentence for what the block should say or do once the update is in.
5. Copy update_id, block_id and author from update, and text from update's text.

An update that changes nothing has empty adds and removes.

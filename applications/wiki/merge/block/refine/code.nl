---
description: Merge for a cell. Write the source of a cell that carries several compatible changes, if their combined meaning is clear.
args:
  block: WikiBlock
  changes: Change[]
  problem?: string
returns: Merged
---
Write the new source of block, a cell, so that it carries every one of changes. They were made to the same base source,
block.text, and they are compatible. block.language says what the source is:

- javascript: the body of a function of `input` (a string). It returns a value of type block.returns.
- natlang: instructions for a small model that takes `input` (a string) and returns block.returns.

Write it in these steps.

1. Start from block.text.
2. Go through changes in order of update_id. Put each one's adds into the source and take out its removes. Changes in
   different statements, branches or sentences go in side by side.
3. Decide whether the combined meaning is clear. It is clear when the changes work on different parts of the source, or
   work on the same part and one reading of the two together is the only reasonable one. It is not clear when two
   changes could be combined in more than one way with different results.
4. Check the source: javascript is a complete function body that returns a block.returns value on every path; natlang
   names its input and says what to return.

When the meaning is clear, return { text: the source, clear: true, reason: "" }. Otherwise return
{ text: block.text, clear: false, reason: the two readings }.

problem, when given, says what an earlier source failed to carry; write a source that carries it.

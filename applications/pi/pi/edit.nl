---
description: Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit. intent says what the change is meant to do.
args:
  path: string
  edits: Edit[]
  intent: string
returns: string
model: small
---
Change the file at path as pi's edit tool does, then check the change against intent.

Read the file (files.read; when there is none, answer `Error: no file at <path>`). Find each edit's oldText in the
original text, every one in the original and not after the others are applied. Answer with an error and change
nothing when:
- edits is empty: `Error: edit needs at least one edits[] entry`;
- an oldText is empty or does not occur: `Error: Could not find the exact text in <path>: <oldText as JSON, its
  first 120 characters>`;
- an oldText occurs more than once: `Error: The text is not unique in <path>; include more context: <the same>`;
- two of the matched regions overlap: `Error: Two edits overlap; merge them into one edit`.
Otherwise replace every matched region with its newText, in a single pass over the original, keeping everything
else byte for byte, and write the result (files.write). Answer `Edited <path>: N replacements` (`1 replacement`).

Then make a unified diff of the change, with --- a/<path> and +++ b/<path> headers, and call decide(review,
intent, diff) with its first 2000 lines. When its answer is unintended or incomplete with probability at least
0.6, add a line to the answer: `[Edit review: this change looks broader than what you said you would do (p=P).
Check the diff.]` for unintended, `[Edit review: this change looks incomplete for what you said you would do (p=P).
Check the diff.]` for incomplete, P being that probability with two decimals.

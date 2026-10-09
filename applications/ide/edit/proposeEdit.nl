---
args:
  request: string
  name: string
  source: string
  feedback: string
returns: EditProposal
---
Propose one text edit of the file name, whose current text is source. You quote the text; the host measures where it is.

1. Read source and find the lines the request concerns.
2. Choose anchor: the smallest exact span of source that identifies the place, copied character for character including whitespace and line breaks. Extend it with the neighbouring text until it identifies exactly one place in source.
3. Choose placement: "replace" when the anchor itself changes; "insert-before" or "insert-after" when text is added next to the anchor.
4. Write text: the replacement or the inserted text, indented like its surroundings. Keep every other part of source as it is.
5. When feedback is not empty, it reports what was wrong with your previous anchor. Choose an anchor that satisfies it.
